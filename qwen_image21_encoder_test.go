package gopherllm

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestQI21TextEncoderTransformersReference(t *testing.T) {
	var fixture struct {
		Config    json.RawMessage `json:"config"`
		Tokenizer json.RawMessage `json:"tokenizer"`
		Prompt    string          `json:"prompt"`
		Tokens    []uint32        `json:"tokens"`
		Drop      int             `json:"drop"`
	}
	data, err := os.ReadFile("testdata/qwen_image21_encoder.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile("testdata/qwen_image21_encoder.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err = os.WriteFile(filepath.Join(dir, "model.safetensors"), data, 0600); err != nil {
		t.Fatal(err)
	}
	weights, err := qi21OpenWeights(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	defer weights.close()
	encoder, err := newQI21TextEncoder(fixture.Config, fixture.Tokenizer, weights)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := encoder.tokenizer.Encode(qi21SystemPrefix + "<|im_start|>user\n" + fixture.Prompt + "<|im_end|>\n<|im_start|>assistant\n")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tokens, fixture.Tokens) || encoder.drop != fixture.Drop {
		t.Fatalf("template tokens or prefix drop differ: %v, drop %d", tokens, encoder.drop)
	}
	got, count, err := encoder.Encode(context.Background(), fixture.Prompt)
	if err != nil {
		t.Fatal(err)
	}
	want, err := weights.floats("reference.hidden", len(tokens)-fixture.Drop, encoder.config.Dim)
	if err != nil {
		t.Fatal(err)
	}
	if count != len(tokens)-fixture.Drop || len(got) != len(want) {
		t.Fatalf("wrong conditioning shape: %d x %d", count, len(got))
	}
	var maximum float64
	for i, v := range want {
		delta := math.Abs(float64(got[i] - v))
		maximum = max(maximum, delta)
		if !finite32(got[i]) || delta > 2e-5+2e-4*math.Abs(float64(v)) {
			t.Fatalf("hidden[%d]=%g want %g (delta %g)", i, got[i], v, delta)
		}
	}
	t.Logf("pre-final-norm conditioning: %d rows, max absolute error %.9g", count, maximum)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err = encoder.Encode(ctx, "ignored"); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation lost: %v", err)
	}
	if _, _, err = encoder.Encode(context.Background(), "<|image_pad|>"); err == nil {
		t.Fatal("vision placeholder accepted by text-only encoder")
	}
	encoder.config.NLayers, encoder.config.KVDim = 128, 65536
	if _, _, err = encoder.Encode(context.Background(), "hello"); err == nil || !strings.Contains(err.Error(), "cache exceeds") {
		t.Fatalf("oversized prompt cache = %v", err)
	}
	encoder.config.MaxSeqLen = 4
	if _, _, err = encoder.Encode(context.Background(), "hello"); err == nil {
		t.Fatal("oversized prompt accepted")
	}
}
