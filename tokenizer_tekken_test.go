package gopherllm

import (
	"reflect"
	"testing"
)

func TestPretokenizeTekken(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"Hello world", []string{"Hello", " world"}},
		{"abc123", []string{"abc", "1", "2", "3"}},
		{"iPhone", []string{"i", "Phone"}},
		{"HELLO", []string{"HELLO"}},
		{" 42", []string{" ", "4", "2"}},
		{"a\n\nb", []string{"a", "\n\n", "b"}},
		{"Hello, World!", []string{"Hello", ",", " World", "!"}},
		{"hi ", []string{"hi", " "}},
		{"  ", []string{"  "}},
		{"3.14", []string{"3", ".", "1", "4"}},
	}
	for _, c := range cases {
		got := pretokenizeTekken(c.in)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("pretokenizeTekken(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestPretokenizeQwen35(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"abc123", []string{"abc", "1", "2", "3"}},
		{"don't I'LL", []string{"don", "'t", " I", "'LL"}},
		{"cafe\u0301 42", []string{"cafe\u0301", " ", "4", "2"}},
		{"a\n\nb", []string{"a", "\n\n", "b"}},
	}
	for _, tc := range cases {
		if got := pretokenizeQwen35(tc.in); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("pretokenizeQwen35(%q) = %#v, want %#v", tc.in, got, tc.want)
		}
	}
}

func TestTokenizerModeDetectsTekken(t *testing.T) {
	// A Tekken GGUF whose tokenizer.ggml.model is not literally "gpt2" must
	// still be classified as GPT2BPE — otherwise it falls through to
	// SentencePiece and mis-tokenizes.
	meta := map[string]MetaValue{
		"tokenizer.ggml.tokens": {Kind: "arr", Value: []string{"<unk>", "<s>", "</s>"}},
		"tokenizer.ggml.model":  {Kind: "str", Value: "llama"},
		"tokenizer.ggml.pre":    {Kind: "str", Value: "tekken"},
	}
	tok, err := TokenizerFromMetadata(meta)
	if err != nil {
		t.Fatalf("TokenizerFromMetadata: %v", err)
	}
	if tok.Mode != TokenizerGPT2BPE {
		t.Fatalf("Mode = %v, want TokenizerGPT2BPE for a tekken pre-tokenizer", tok.Mode)
	}
}

func TestPretokenizeDispatch(t *testing.T) {
	tek := &Tokenizer{Pre: "tekken"}
	if got := tek.pretokenize("a1"); !reflect.DeepEqual(got, []string{"a", "1"}) {
		t.Fatalf("tekken dispatch = %q, want [a 1]", got)
	}
	// Non-tekken GPT-2 keeps grouped digits.
	gpt := &Tokenizer{Pre: "qwen2"}
	if got := gpt.pretokenize("a12"); !reflect.DeepEqual(got, []string{"a", "12"}) {
		t.Fatalf("gpt2 dispatch = %q, want [a 12]", got)
	}
	qwen35 := &Tokenizer{Pre: "qwen35"}
	if got := qwen35.pretokenize("a12"); !reflect.DeepEqual(got, []string{"a", "1", "2"}) {
		t.Fatalf("qwen35 dispatch = %q, want [a 1 2]", got)
	}
}
