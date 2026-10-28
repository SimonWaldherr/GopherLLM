//go:build darwin && cgo && metal

package engine

import (
	"fmt"
	"math"
	"testing"
)

// BenchmarkQwen35MetalMixer27B isolates a complete recurrent mixer with the
// 27B projection geometry. Both schedules batch Out identically; only the
// independent input projections change. The fixture retains one layer and
// repeats encoded rows, avoiding a complete checkpoint or F32 weight bank.
func BenchmarkQwen35MetalMixer27B(b *testing.B) {
	if !MetalAvailable() {
		b.Skip(MetalError())
	}
	if !q8ActivationsAvailable() {
		b.Skip("Q8 activation instructions unavailable")
	}
	setPerformanceThreads(b, 8)
	withQ8Activations(true, func() {
		cfg := Config{Dim: 5120, SSMInner: 6144, SSMHeads: 48, SSMGroups: 16,
			SSMState: 128, SSMConv: 4, RMSNormEps: 1e-6}
		channels := cfg.SSMInner + 2*cfg.SSMGroups*cfg.SSMState
		w := Qwen35DeltaNetWeights{
			QKVConv:    qwen35InputWeight(b, GGMLTypeQ6_K, channels, cfg.Dim),
			Gate:       qwen35InputWeight(b, GGMLTypeQ4_K, cfg.SSMInner, cfg.Dim),
			AlphaProj:  qwen35InputWeight(b, GGMLTypeQ4_K, cfg.SSMHeads, cfg.Dim),
			BetaProj:   qwen35InputWeight(b, GGMLTypeQ4_K, cfg.SSMHeads, cfg.Dim),
			Out:        qwen35InputWeight(b, GGMLTypeQ4_K, cfg.Dim, cfg.SSMInner),
			ConvKernel: Weight{F32: randomExpertInput(channels*cfg.SSMConv, 44210)},
			A:          make([]float32, cfg.SSMHeads), DTBias: randomExpertInput(cfg.SSMHeads, 44211),
			Norm: make([]float32, cfg.SSMState),
		}
		for i := range w.ConvKernel.F32 {
			w.ConvKernel.F32[i] *= .004
		}
		for i := range w.A {
			w.A[i] = -.2 - float32(i)*.003
		}
		for i := range w.Norm {
			w.Norm[i] = 1
		}
		w.QKVConv.Metal = prepareMetalWeight(w.QKVConv.Raw, w.QKVConv.Type, channels, cfg.Dim, false)
		if !metalWeightUsesDirect(w.QKVConv.Metal) {
			b.Fatalf("fixture requires direct Metal QKV: %s", MetalError())
		}
		b.Cleanup(func() { releaseMetalWeight(w.QKVConv.Metal) })
		if !qwen35MetalBatchCPUInput(w.Gate) || !qwen35MetalBatchCPUInput(w.AlphaProj) ||
			!qwen35MetalBatchCPUInput(w.BetaProj) || !qwen35MetalBatchCPUOut(w.Out) {
			b.Fatal("fixture requires CPU-Q8 gate, alpha/beta and Out projections")
		}
		initial := newQwen35Cache(cfg, 1)
		copy(initial.Conv, randomExpertInput(len(initial.Conv), 44212))
		copy(initial.State, randomExpertInput(len(initial.State), 44213))
		for i := range initial.Conv {
			initial.Conv[i] *= .004
		}
		for i := range initial.State {
			initial.State[i] *= .005
		}
		checkBits := func(name string, got, want []float32) {
			b.Helper()
			if len(got) != len(want) {
				b.Fatalf("%s lengths differ: %d/%d", name, len(got), len(want))
			}
			for i, value := range got {
				if !finite32(value) || math.Float32bits(value) != math.Float32bits(want[i]) {
					b.Fatalf("%s[%d] bits %08x/%08x", name, i, math.Float32bits(value), math.Float32bits(want[i]))
				}
			}
		}
		for _, p := range []int{4, 16, 32} {
			type workspace struct {
				state *Qwen35Cache
				buf   DecodeBuffer
				batch qwen35BatchScratch
			}
			newWorkspace := func() *workspace {
				s := &workspace{state: initial.snapshot()}
				prepareQwen35InputRows(&s.batch, cfg, p)
				copy(s.batch.XNFlat, randomExpertInput(len(s.batch.XNFlat), 44214))
				reuseBatchViews(&s.batch.MixedFlat, &s.batch.Mixed, p, cfg.SSMInner)
				reuseBatchViews(&s.batch.ProjFlat, &s.batch.Proj, p, cfg.Dim)
				return s
			}
			run := func(t testing.TB, s *workspace, bulk bool) {
				var haveQKV, haveGate, haveAB bool
				if bulk {
					haveQKV, haveGate, haveAB = qwen35MetalBatchDeltaInputs(w, s.batch.XN, &s.batch)
					if !haveQKV || !haveGate || !haveAB {
						t.Fatal("fixture did not batch all independent mixer inputs")
					}
				}
				for token, x := range s.batch.XN {
					if bulk {
						qkv, gate, alpha, beta := qwen35DeltaNetBatchInputs(w, x, &s.buf,
							s.batch.QKV[token], s.batch.Gate[token], s.batch.Alpha[token], s.batch.Beta[token],
							haveQKV, haveGate, haveAB)
						qwen35DeltaNetStep(cfg, w, s.state, qkv, gate, alpha, beta, 0, &s.buf)
					} else {
						qwen35DeltaNetMix(cfg, w, s.state, x, 0, &s.buf)
					}
					copy(s.batch.Mixed[token], s.buf.MambaY)
				}
				matvecBatchSlabs(w.Out, s.batch.Mixed, s.batch.Proj, s.batch.MixedFlat, s.batch.ProjFlat)
			}
			serial, bulk := newWorkspace(), newWorkspace()
			run(b, serial, false)
			run(b, bulk, true)
			checkBits("mixer output", bulk.batch.MixedFlat, serial.batch.MixedFlat)
			checkBits("Out projection", bulk.batch.ProjFlat, serial.batch.ProjFlat)
			checkBits("convolution", bulk.state.Conv, serial.state.Conv)
			checkBits("recurrent state", bulk.state.State, serial.state.State)
			b.Run(fmt.Sprintf("positions=%d", p), func(b *testing.B) {
				for _, mode := range []bool{false, true} {
					b.Run(fmt.Sprintf("bulk=%t", mode), func(b *testing.B) {
						s := newWorkspace()
						run(b, s, mode) // Prepare pooled tasks and GPU workspace before timing.
						if !s.state.restore(initial) {
							b.Fatal("cannot restore common initial recurrent state")
						}
						b.ReportAllocs()
						b.ResetTimer()
						for b.Loop() {
							run(b, s, mode)
						}
					})
				}
			})
		}
	})
}
