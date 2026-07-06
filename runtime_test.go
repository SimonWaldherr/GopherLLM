package gopherllm

import (
	"math"
	"testing"
)

func TestMeanPoolScalesBySampleCount(t *testing.T) {
	v := []float32{6, -3, 9}
	meanPoolInPlace(v, 3)
	want := []float32{2, -1, 3}
	for i := range want {
		if v[i] != want[i] {
			t.Fatalf("v[%d] = %v, want %v", i, v[i], want[i])
		}
	}
}

func TestL2NormalizeProducesUnitVector(t *testing.T) {
	v := []float32{3, 4}
	l2NormalizeInPlace(v)
	norm := math.Sqrt(float64(v[0]*v[0] + v[1]*v[1]))
	if math.Abs(norm-1) > 1e-6 {
		t.Fatalf("norm = %v, want 1", norm)
	}
}

func TestCosineSimilarityRejectsInvalidInputs(t *testing.T) {
	if _, err := CosineSimilarity(nil, nil); err == nil {
		t.Fatal("empty vectors should fail")
	}
	if _, err := CosineSimilarity([]float32{1}, []float32{1, 2}); err == nil {
		t.Fatal("dimension mismatch should fail")
	}
	if _, err := CosineSimilarity([]float32{0, 0}, []float32{1, 2}); err == nil {
		t.Fatal("zero-norm vector should fail")
	}
}

func TestRopeInterleavedArchitectureSelection(t *testing.T) {
	for _, arch := range []string{"llama", "llama2", "llama3", "mistral", "mistral3", "mixtral", "ministral"} {
		if !ropeInterleaved(arch) {
			t.Fatalf("ropeInterleaved(%q) = false, want true", arch)
		}
	}
	for _, arch := range []string{"qwen2", "phi3", "granite", "deepseek2"} {
		if ropeInterleaved(arch) {
			t.Fatalf("ropeInterleaved(%q) = true, want false", arch)
		}
	}
}

func TestGenerationOptionsRejectsInvalidFloatsAndTopK(t *testing.T) {
	opts := DefaultGenerationOptions()
	opts.Sampler.Temperature = float32(math.Inf(1))
	if err := opts.Validate(); err == nil {
		t.Fatal("infinite temperature should fail")
	}

	opts = DefaultGenerationOptions()
	opts.Sampler.TopP = float32(math.Inf(1))
	if err := opts.Validate(); err == nil {
		t.Fatal("infinite top_p should fail")
	}

	opts = DefaultGenerationOptions()
	opts.Sampler.RepeatPenalty = float32(math.Inf(1))
	if err := opts.Validate(); err == nil {
		t.Fatal("infinite repeat_penalty should fail")
	}

	opts = DefaultGenerationOptions()
	opts.Sampler.TopK = -1
	if err := opts.Validate(); err == nil {
		t.Fatal("negative top_k should fail")
	}
}

func TestPrefillChunkSizeEnvOverride(t *testing.T) {
	t.Setenv("GOPHERLLM_PREFILL_CHUNK", "")
	if got := prefillChunkSize(Config{Dim: 4096, HiddenDim: 14336}); got != 32 {
		t.Fatalf("large-model default chunk = %d, want 32", got)
	}
	if got := prefillChunkSize(Config{Dim: 3072, HiddenDim: 9216}); got != 128 {
		t.Fatalf("small-model default chunk = %d, want 128", got)
	}
	t.Setenv("GOPHERLLM_PREFILL_CHUNK", "64")
	if got := prefillChunkSize(Config{Dim: 3072, HiddenDim: 9216}); got != 64 {
		t.Fatalf("override chunk = %d, want 64", got)
	}
	t.Setenv("GOPHERLLM_PREFILL_CHUNK", "999")
	if got := prefillChunkSize(Config{}); got != 256 {
		t.Fatalf("clamped chunk = %d, want 256", got)
	}
	t.Setenv("GOPHERLLM_PREFILL_CHUNK", "nope")
	if got := prefillChunkSize(Config{}); got != 32 {
		t.Fatalf("invalid chunk = %d, want 32", got)
	}
}

func TestValidUTF8PrefixLenKeepsIncompleteRuneBuffered(t *testing.T) {
	b := []byte{'H', 'i', ' ', 0xe2, 0x82}
	if got := validUTF8PrefixLen(b); got != 3 {
		t.Fatalf("prefix length = %d, want 3", got)
	}
}

func TestValidUTF8PrefixLenAcceptsCompletedRune(t *testing.T) {
	b := []byte{'H', 'i', ' ', 0xe2, 0x82, 0xac}
	if got := validUTF8PrefixLen(b); got != len(b) {
		t.Fatalf("prefix length = %d, want %d", got, len(b))
	}
}
