package engine

import (
	"math"
	"os"
	"testing"
)

// The golden values use an independent NumPy FP32 forward over the original
// LittleBitLLM/littlebit-qwen3-8b-0.55bpw safetensors. The reference expands
// each packed sign matrix with unpackbits(bitorder="little"), widens the BF16
// scales to FP32 and computes the two products independently. It performs
// the full 36-layer Qwen3 forward for the last-token logits below.
func TestLittleBitPublishedCheckpointParity(t *testing.T) {
	path := os.Getenv("GOPHERLLM_LITTLEBIT_TEST_GGUF")
	if path == "" {
		t.Skip("set GOPHERLLM_LITTLEBIT_TEST_GGUF to the converted Qwen3-8B LittleBit-2 checkpoint")
	}
	useMetal := os.Getenv("GOPHERLLM_LITTLEBIT_TEST_METAL") == "1"
	if useMetal && !MetalAvailable() {
		t.Fatal("Metal requested but unavailable")
	}
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: useMetal, OutOfCore: !useMetal})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	c := r.Config()
	if r.Architecture() != "qwen3" || c.NLayers != 36 || c.VocabSize != 151936 {
		t.Fatalf("expected the public Qwen3-8B checkpoint, got %s with %d layers and vocabulary %d", r.Architecture(), c.NLayers, c.VocabSize)
	}

	type goldenValue struct {
		index int
		value float32
	}
	check := func(got []float32, want []goldenValue, tolerance float64) {
		t.Helper()
		for _, item := range want {
			if value := got[item.index]; !finite32(value) || math.Abs(float64(value-item.value)) > tolerance {
				t.Errorf("output %d = %.9g, reference %.9g (absolute tolerance %g)", item.index, value, item.value, tolerance)
			}
		}
	}
	first := r.standard.Layers[0]
	for _, tc := range []struct {
		name string
		w    Weight
		rank int
		top  uint32
		want []goldenValue
	}{
		{"q", first.WQ, 544, 3739, []goldenValue{
			{0, -0.03238495}, {1, 0.16480744}, {2, -0.06279524}, {31, 0.63223839},
			{127, -1.10547888}, {1024, 1.13903797}, {2048, 0.93974483}, {4095, 0.60584301},
		}},
		{"down", first.W2, 824, 1838, []goldenValue{
			{0, 1.12851810}, {1, -3.04956269}, {2, 0.37802804}, {31, 0.16761720},
			{127, -1.46643972}, {1024, 1.34711635}, {2048, 1.71410632}, {4095, -0.06939036},
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			factors := tc.w.LittleBit
			if factors == nil || factors.Residual == nil || factors.Main.Rank != tc.rank || factors.Residual.Rank != tc.rank {
				t.Fatalf("expected two LittleBit paths of rank %d", tc.rank)
			}
			if useMetal && factors.Metal == nil {
				t.Fatal("projection did not acquire Metal storage")
			}
			x := make([]float32, tc.w.Cols)
			for i := range x {
				x[i] = float32((i*13+7)%127-63) / 64
			}
			y := tc.w.Matvec(x)
			check(y, tc.want, 2e-5)
			if token := argmaxFiniteToken(y); token != tc.top {
				t.Errorf("argmax %d, want %d", token, tc.top)
			}
		})
	}

	tokens := []uint32{16, 17, 18, 19}
	want := []goldenValue{{13, 18.02217102}, {18, 17.61205673}, {20, 17.23637772}, {19, 17.18972969}, {16, 17.12265396}}
	for _, batch := range []bool{false, true} {
		name := "decode"
		if batch {
			name = "prefill"
		}
		t.Run(name, func(t *testing.T) {
			k, v, head, kvHeads, value := r.cacheDims()
			cache := NewKVCache(c.NLayers, k, v, len(tokens))
			scratch := NewDecodeBuffer(c, head, kvHeads, value)
			var logits []float32
			if batch {
				ForwardBatchInto(c, r.standard, cache, scratch, tokens, 0, true, &logits)
			} else {
				for i, token := range tokens {
					ForwardInto(c, r.standard, cache, scratch, token, i, &logits)
				}
			}
			if len(logits) != c.VocabSize {
				t.Fatalf("received %d logits, want %d", len(logits), c.VocabSize)
			}
			check(logits, want, 2e-3)
			if token := argmaxFiniteToken(logits); token != uint32(want[0].index) {
				t.Errorf("argmax %d, want %d", token, want[0].index)
			}
		})
	}
}
