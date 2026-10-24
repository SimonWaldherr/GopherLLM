//go:build darwin && cgo && metal

package gopherllm

import "testing"

// A wide QKV projection bypasses CPU pair fusion even with a narrow gate.
// Compare the loader policy against the original accelerated mixer over
// several recurrent updates, including the Q8 activation rounding boundary.
func TestQwen35MoELoadPolicyPreservesMetalMixer(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	cfg := Config{Arch: "qwen35moe", Dim: 256, NLayers: 1, SSMInner: 4096,
		SSMHeads: 64, SSMGroups: 32, SSMState: 64, SSMConv: 4, RMSNormEps: 1e-5}
	channels := cfg.SSMInner + 2*cfg.SSMGroups*cfg.SSMState
	for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K, GGMLTypeQ8_0} {
		t.Run(typ.String(), func(t *testing.T) {
			w := Qwen35DeltaNetWeights{
				QKVConv:    qwen35InputWeight(t, typ, channels, cfg.Dim),
				Gate:       qwen35InputWeight(t, typ, cfg.SSMInner, cfg.Dim),
				AlphaProj:  qwen35InputWeight(t, typ, cfg.SSMHeads, cfg.Dim),
				BetaProj:   qwen35InputWeight(t, typ, cfg.SSMHeads, cfg.Dim),
				ConvKernel: Weight{F32: smallWeights(channels*cfg.SSMConv, 4701), Rows: channels, Cols: cfg.SSMConv},
				A:          make([]float32, cfg.SSMHeads), DTBias: smallWeights(cfg.SSMHeads, 4702), Norm: make([]float32, cfg.SSMState),
			}
			for i := range w.A {
				w.A[i] = -.3
			}
			for i := range w.Norm {
				w.Norm[i] = 1
			}
			names := [...]string{"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"}
			source := [...]Weight{w.QKVConv, w.Gate, w.AlphaProj, w.BetaProj}
			tensors := make(map[string]TensorInfo)
			for i, name := range names {
				tensors[name] = TensorInfo{Name: name, DType: typ, Dims: []uint64{uint64(source[i].Cols), uint64(source[i].Rows)}}
			}
			cpu := qwen35CPUOnlyWeights(cfg, tensors)
			load := func(policy bool) Qwen35DeltaNetWeights {
				loaded := w
				fields := [...]*Weight{&loaded.QKVConv, &loaded.Gate, &loaded.AlphaProj, &loaded.BetaProj}
				for i, name := range names {
					var err error
					*fields[i], err = loadWeight(source[i].Raw, 0, name, tensors, nil, false, false, false, !policy || !cpu[name])
					if err != nil {
						t.Fatal(err)
					}
				}
				t.Cleanup(func() {
					var releaser weightResourceReleaser
					for _, field := range fields {
						releaser.release(field)
					}
				})
				return loaded
			}
			actual, reference := load(true), load(false)
			if !metalWeightUsesDirect(actual.QKVConv.Metal) || !metalWeightUsesDirect(reference.QKVConv.Metal) {
				t.Fatal("direct QKV projection lost its Metal view")
			}
			for _, q8 := range []bool{false, true} {
				if q8 && !q8ActivationsAvailable() {
					continue
				}
				withQ8Activations(q8, func() {
					state, expected := newQwen35Cache(cfg, 1), newQwen35Cache(cfg, 1)
					buf, rb := &DecodeBuffer{}, &DecodeBuffer{}
					for position := range 4 {
						x := randomExpertInput(cfg.Dim, int64(4710+position))
						qwen35DeltaNetMix(cfg, actual, state, x, 0, buf)
						qwen35DeltaNetMix(cfg, reference, expected, x, 0, rb)
						requireMoEPrefillExact(t, buf.MambaIn, rb.MambaIn)
						requireMoEPrefillExact(t, state.Conv, expected.Conv)
						requireMoEPrefillExact(t, state.State, expected.State)
						requireMoEPrefillExact(t, buf.MambaY, rb.MambaY)
					}
				})
			}
		})
	}
}
