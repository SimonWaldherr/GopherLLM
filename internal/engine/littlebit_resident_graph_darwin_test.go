//go:build darwin && cgo && metal

package engine

import (
	"context"
	"maps"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestQwen3LittleBitResidentPrefillParity(t *testing.T) {
	r, _, useMetal, _ := openQwen3ComparisonModel(t)
	if !useMetal || !metalDenseLittleBitModel(r.standard) {
		t.Skip("requires Metal and a LittleBit Qwen3-8B checkpoint")
	}
	t.Setenv("GOPHERLLM_METAL_LITTLEBIT_RESIDENT", "1")
	t.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "1")
	t.Setenv("GOPHERLLM_METAL_DENSE_PREFILL", "1")
	prompt := r.tok.EncodeWithoutBOS(strings.Repeat("Air scatters blue light more than red light. ", 32))
	if len(prompt) < 128 {
		t.Fatalf("comparison prompt has only %d tokens", len(prompt))
	}
	prompt = prompt[:128]
	cache, buf := r.generationWorkspace(len(prompt))
	t.Setenv("GOPHERLLM_METAL_LITTLEBIT_PREFILL", "0")
	var grouped []float32
	if err := forwardBatchIntoContext(context.Background(), r.config, r.standard, cache, buf, prompt, 0, true, &grouped, nil); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOPHERLLM_METAL_LITTLEBIT_PREFILL", "1")
	var resident []float32
	if err := forwardBatchIntoContext(context.Background(), r.config, r.standard, cache, buf, prompt, 0, true, &resident, nil); err != nil {
		t.Fatal(err)
	}
	if buf.metalDense == nil || buf.metalDense.decoder.PrefillScratchBytes() == 0 {
		t.Fatal("eligible LittleBit prefill did not take the resident path")
	}
	if len(grouped) != r.config.VocabSize || len(resident) != len(grouped) {
		t.Fatalf("unexpected vocabulary sizes grouped=%d resident=%d", len(grouped), len(resident))
	}
	var maxAbs float64
	for i, want := range grouped {
		got := resident[i]
		abs := math.Abs(float64(got - want))
		if !finite32(want) || !finite32(got) || abs > 2e-3 {
			t.Fatalf("logit %d resident=%g grouped=%g absolute error=%g", i, got, want, abs)
		}
		maxAbs = max(maxAbs, abs)
	}
	if argmaxFiniteToken(resident) != argmaxFiniteToken(grouped) {
		t.Fatal("greedy token changed")
	}
	t.Logf("compared %d logits, max absolute difference %g, resident prefill scratch %d bytes",
		len(grouped), maxAbs, buf.metalDense.decoder.PrefillScratchBytes())
}

func TestLittleBitMetalResidentGraph(t *testing.T) {
	testLittleBitMetalResidentGraph(t, false)
}

func TestLittleBitMetalResidentPrefill(t *testing.T) {
	testLittleBitMetalResidentGraph(t, true)
}

func testLittleBitMetalResidentGraph(t *testing.T, wholePrefill bool) {
	t.Helper()
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	if os.Getenv("GOPHERLLM_METAL_LITTLEBIT_RESIDENT") == "0" || !metalDenseDecodeEnabled {
		t.Skip("resident Metal decode disabled")
	}
	if wholePrefill {
		t.Setenv("GOPHERLLM_METAL_LITTLEBIT_PREFILL", "")
		t.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "1")
		t.Setenv("GOPHERLLM_METAL_DENSE_PREFILL", "1")
	}
	for _, arch := range []string{"llama", "qwen3"} {
		for _, shared := range []string{"0", "1"} {
			t.Run(arch+"/sharedKV="+shared, func(t *testing.T) {
				t.Setenv("GOPHERLLM_METAL_DENSE_SHARED_KV", shared)
				f := newLittleBitModelFixture(t, 256, 288, 2, 2, 1, 35, 7, 2, true)
				if arch == "qwen3" {
					f.meta["general.architecture"] = MetaValue{Kind: "str", Value: arch}
					for key, value := range maps.Clone(f.meta) {
						if suffix, ok := strings.CutPrefix(key, "llama."); ok {
							delete(f.meta, key)
							f.meta[arch+"."+suffix] = value
						}
					}
					for _, layer := range []string{"blk.0.", "blk.1."} {
						for _, name := range []string{"attn_q_norm.weight", "attn_k_norm.weight"} {
							values := onesF32(128)
							for i := range values {
								values[i] += float32(i%9-4) * 0.03125
							}
							v := ggufTensor{name: layer + name, dims: []uint64{128}, dtype: GGMLTypeF32, data: f32Bytes(values)}
							f.packed, f.dense = append(f.packed, v), append(f.dense, v)
						}
					}
				}
				path := filepath.Join(t.TempDir(), "resident.gguf")
				if err := os.WriteFile(path, writeLittleBitFixture(t, f.meta, f.packed), 0600); err != nil {
					t.Fatal(err)
				}
				got, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true})
				if err != nil {
					t.Fatal(err)
				}
				defer got.Close()
				want := f.reference(t)

				c := got.Config()
				kDim, vDim, head, kvHeads, value := got.cacheDims()
				gotK, wantK := NewKVCache(c.NLayers, kDim, vDim, 128), NewKVCache(c.NLayers, kDim, vDim, 128)
				gotB, wantB := NewDecodeBuffer(c, head, kvHeads, value), NewDecodeBuffer(c, head, kvHeads, value)
				defer func() { gotB.metalDense.close() }()
				prefix := []uint32{1, 3, 4, 5}
				if wholePrefill {
					for i := len(prefix); i < 31; i++ {
						prefix = append(prefix, uint32(i%c.VocabSize))
					}
				}
				var gotLogits, wantLogits []float32
				ForwardBatchInto(c, got.standard, gotK, gotB, prefix, 0, true, &gotLogits)
				if wholePrefill && (gotB.metalDense == nil || gotB.metalDense.decoder.PrefillScratchBytes() == 0) {
					t.Fatal("eligible LittleBit prefill did not take the resident path")
				}
				for pos, token := range prefix {
					ForwardInto(c, want.standard, wantK, wantB, token, pos, &wantLogits)
				}
				check := func(label string, got, want []float32) {
					t.Helper()
					if len(got) != len(want) {
						t.Fatalf("%s length %d, want %d", label, len(got), len(want))
					}
					for i, value := range want {
						if !finite32(got[i]) || math.Abs(float64(got[i]-value)) > 1e-4*math.Max(1, math.Abs(float64(value))) {
							t.Fatalf("%s[%d]=%g, want %g", label, i, got[i], value)
						}
					}
				}
				check("prefill logits", gotLogits, wantLogits)

				// Overwrite the rejected suffix, as speculative verification or a
				// changed chat prefix can. Copy and shared-KV modes must both honor
				// the new position instead of using stale device pages.
				for i, delta := range []int{0, 1, 2, 1, 2, 3} {
					pos, token := len(prefix)+delta, uint32(6+i)
					if !tryMetalDenseDecode(c, got.standard, gotK, gotB, token, pos) {
						t.Fatalf("eligible LittleBit body fell back at position %d", pos)
					}
					ProjectLogitsInto(c, got.standard, gotB, &gotLogits)
					ForwardInto(c, want.standard, wantK, wantB, token, pos, &wantLogits)
					check("decode logits", gotLogits, wantLogits)
					for layer := range c.NLayers {
						check("K", gotK.K[layer][:(pos+1)*kDim], wantK.K[layer][:(pos+1)*kDim])
						check("V", gotK.V[layer][:(pos+1)*vDim], wantK.V[layer][:(pos+1)*vDim])
					}
				}
				if wholePrefill {
					rewind := len(prefix) - 3
					suffix := prefix[:19]
					ForwardBatchInto(c, got.standard, gotK, gotB, suffix, rewind, true, &gotLogits)
					for i, token := range suffix {
						ForwardInto(c, want.standard, wantK, wantB, token, rewind+i, &wantLogits)
					}
					check("rewound prefill logits", gotLogits, wantLogits)
					for layer := range c.NLayers {
						end := rewind + len(suffix)
						check("rewound K", gotK.K[layer][:end*kDim], wantK.K[layer][:end*kDim])
						check("rewound V", gotK.V[layer][:end*vDim], wantK.V[layer][:end*vDim])
					}
				}
				if gotB.metalDense == nil || !gotB.metalDense.littleBit || (gotB.metalDense.sharedK[0] != nil) != (shared == "1" && metalKVShareable(gotK.K[0]) && metalKVShareable(gotK.V[0])) {
					t.Fatal("unexpected resident Metal KV storage")
				}
			})
		}
	}
}
