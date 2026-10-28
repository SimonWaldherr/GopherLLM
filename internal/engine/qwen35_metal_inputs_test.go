package engine

import (
	"fmt"
	"slices"
	"testing"
)

func qwen35InputWeight(t testing.TB, typ GGMLType, rows, cols int) Weight {
	t.Helper()
	rowBytes, ok := typ.DataSize(cols)
	if !ok {
		t.Fatal("unsupported fixture type")
	}
	w := Weight{Type: typ, Rows: rows, Cols: cols, Raw: make([]byte, rows*rowBytes)}
	patterns := make([][]byte, min(rows, 32))
	for i := range patterns {
		x := randomExpertInput(cols, int64(44000+i))
		switch typ {
		case GGMLTypeQ4_K:
			patterns[i] = QuantizeRowQ4K(x, cols)
		case GGMLTypeQ6_K:
			patterns[i] = QuantizeRowQ6K(x, cols)
		case GGMLTypeQ8_0:
			patterns[i] = QuantizeRowQ8_0(x, cols)
		default:
			t.Fatal("unsupported fixture encoder")
		}
	}
	for row := range rows {
		copy(w.Raw[row*rowBytes:], patterns[row%len(patterns)])
	}
	return w
}

func prepareQwen35InputRows(b *qwen35BatchScratch, cfg Config, p int) {
	reuseBatchViews(&b.XNFlat, &b.XN, p, cfg.Dim)
	reuseBatchViews(&b.QKVFlat, &b.QKV, p, cfg.SSMInner+2*cfg.SSMGroups*cfg.SSMState)
	reuseBatchViews(&b.GateFlat, &b.Gate, p, cfg.SSMInner)
	reuseBatchViews(&b.AlphaFlat, &b.Alpha, p, cfg.SSMHeads)
	reuseBatchViews(&b.BetaFlat, &b.Beta, p, cfg.SSMHeads)
}

func TestQwen35MetalCPUInputsPreserveProjectionAndStateBits(t *testing.T) {
	if !q8ActivationsAvailable() {
		t.Skip("Q8 activation instructions unavailable")
	}
	withQ8Activations(true, func() {
		for _, types := range [][2]GGMLType{{GGMLTypeQ4_K, GGMLTypeQ4_K}, {GGMLTypeQ6_K, GGMLTypeQ4_K}, {GGMLTypeQ8_0, GGMLTypeQ8_0}} {
			cfg, weights := qwen35BatchFixture(true)
			w := weights.Layers[0].DeltaNet
			w.QKVConv = qwen35InputWeight(t, types[0], cfg.SSMInner+2*cfg.SSMGroups*cfg.SSMState, cfg.Dim)
			w.Gate = qwen35InputWeight(t, types[1], cfg.SSMInner, cfg.Dim)
			w.AlphaProj = qwen35InputWeight(t, GGMLTypeQ4_K, cfg.SSMHeads, cfg.Dim)
			w.BetaProj = qwen35InputWeight(t, GGMLTypeQ4_K, cfg.SSMHeads, cfg.Dim)
			for _, p := range []int{1, 2, 4, 8, 17, 32} {
				t.Run(fmt.Sprintf("%s+%s/p=%d", types[0], types[1], p), func(t *testing.T) {
					var b qwen35BatchScratch
					prepareQwen35InputRows(&b, cfg, p)
					copy(b.XNFlat, randomExpertInput(len(b.XNFlat), 44040))
					haveQKV, haveGate, haveAB := qwen35MetalBatchDeltaInputs(w, b.XN, &b)
					if haveQKV != (p > 1) || haveGate != (p > 1) || haveAB != (p > 1) {
						t.Fatalf("unexpected batching decisions: %t %t %t", haveQKV, haveGate, haveAB)
					}
					state, reference := newQwen35Cache(cfg, 1), newQwen35Cache(cfg, 1)
					buf, rb := &DecodeBuffer{}, &DecodeBuffer{}
					for i, x := range b.XN {
						qkv, gate, alpha, beta := qwen35DeltaNetBatchInputs(w, x, buf,
							b.QKV[i], b.Gate[i], b.Alpha[i], b.Beta[i], haveQKV, haveGate, haveAB)
						qwen35DeltaNetMix(cfg, w, reference, x, 0, rb)
						requireMoEPrefillExact(t, qkv, rb.MambaIn)
						requireMoEPrefillExact(t, gate, rb.MambaZ)
						requireMoEPrefillExact(t, alpha, rb.MambaDT)
						requireMoEPrefillExact(t, beta, rb.MambaBeta)
						qwen35DeltaNetStep(cfg, w, state, qkv, gate, alpha, beta, 0, buf)
						requireMoEPrefillExact(t, state.Conv, reference.Conv)
						requireMoEPrefillExact(t, state.State, reference.State)
						requireMoEPrefillExact(t, buf.MambaY, rb.MambaY)
					}
				})
			}
		}
	})
}

func TestQwen35MetalCPUInputKeepsPreparedAndFloatPaths(t *testing.T) {
	withQ8Activations(true, func() {
		w := qwen35InputWeight(t, GGMLTypeQ4_K, 8, 256)
		if got, want := qwen35MetalBatchCPUInput(w), q8ActivationsAvailable(); got != want {
			t.Fatalf("input admission=%t, want %t with Q8 availability", got, want)
		}
		for _, edit := range []func(*Weight){
			func(w *Weight) { w.Prepared = &PreparedQuantizedWeight{} },
			func(w *Weight) { w.F32 = make([]float32, 8*256) },
			func(w *Weight) { w.Raw = w.Raw[:1] },
			func(w *Weight) { w.Cols = 128 },
		} {
			copyWeight := w
			edit(&copyWeight)
			if qwen35MetalBatchCPUInput(copyWeight) {
				t.Fatal("incompatible projection accepted")
			}
		}
	})
	withQ8Activations(false, func() {
		if qwen35MetalBatchCPUInput(qwen35InputWeight(t, GGMLTypeQ4_K, 8, 256)) {
			t.Fatal("float activation path changed")
		}
	})
}

func BenchmarkQwen35InputGate27B(b *testing.B) {
	if !q8ActivationsAvailable() {
		b.Skip("Q8 activation instructions unavailable")
	}
	w := qwen35InputWeight(b, GGMLTypeQ4_K, 6144, 5120)
	for _, p := range []int{4, 16, 32} {
		for _, bulk := range []bool{false, true} {
			b.Run(fmt.Sprintf("positions=%d/bulk=%t", p, bulk), func(b *testing.B) {
				setPerformanceThreads(b, 8)
				withQ8Activations(true, func() {
					var input, output []float32
					var xs, outs [][]float32
					xs = reuseBatchViews(&input, &xs, p, w.Cols)
					outs = reuseBatchViews(&output, &outs, p, w.Rows)
					copy(input, randomExpertInput(len(input), 44041))
					run := func() {
						if bulk {
							matvecBatchSlabs(w, xs, outs, input, output)
						} else {
							for i := range xs {
								w.MatvecInto(xs[i], &outs[i])
							}
						}
					}
					run()
					reference := slices.Clone(output)
					for i := range xs {
						w.MatvecInto(xs[i], &outs[i])
					}
					requireMoEPrefillExact(b, output, reference)
					b.ReportAllocs()
					b.ResetTimer()
					for b.Loop() {
						run()
					}
				})
			})
		}
	}
}
