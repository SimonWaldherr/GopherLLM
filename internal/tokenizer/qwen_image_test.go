package tokenizer

import (
	"encoding/json"
	"os"
	"slices"
	"testing"
)

func qwenImageReferenceRows(t *testing.T) []struct {
	Text   string
	Pieces []string
	Tokens []uint32
} {
	t.Helper()
	data, err := os.ReadFile("../../testdata/qwen_image21_tokenizer_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var rows []struct {
		Text   string
		Pieces []string
		Tokens []uint32
	}
	if err = json.Unmarshal(data, &rows); err != nil {
		t.Fatal(err)
	}
	return rows
}

func TestQwenImagePretokenizerReference(t *testing.T) {
	for _, row := range qwenImageReferenceRows(t) {
		if got := qwenImagePieces(normalizeNFC(row.Text)); !slices.Equal(got, row.Pieces) {
			t.Fatalf("%q: pieces=%q want %q", row.Text, got, row.Pieces)
		}
	}
}

func TestQwenImageOfficialTokenizer(t *testing.T) {
	path := os.Getenv("GOPHERLLM_QI21_TOKENIZER")
	if path == "" {
		t.Skip("set GOPHERLLM_QI21_TOKENIZER to official processor/tokenizer.json")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := NewQwenImageTokenizer(data)
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range qwenImageReferenceRows(t) {
		got, err := tok.Encode(row.Text)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, row.Tokens) {
			t.Fatalf("%q: ids=%v want %v", row.Text, got, row.Tokens)
		}
	}
}

func TestQwenImageTokenizerRejectsUnknownPipeline(t *testing.T) {
	data, err := os.ReadFile("../../testdata/qwen_image21_encoder.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct{ Tokenizer map[string]any }
	if err = json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	good, err := json.Marshal(fixture.Tokenizer)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := NewQwenImageTokenizer(good)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = tok.Encode(string([]byte{0xff})); err == nil {
		t.Fatal("invalid UTF-8 accepted")
	}
	if err = tok.ValidateVocab(256); err == nil {
		t.Fatal("out-of-range added token accepted")
	}
	fixture.Tokenizer["normalizer"] = map[string]any{"type": "Lowercase"}
	bad, _ := json.Marshal(fixture.Tokenizer)
	if _, err = NewQwenImageTokenizer(bad); err == nil {
		t.Fatal("unsupported normalizer accepted")
	}
}
