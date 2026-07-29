package gopherllm

import (
	"reflect"
	"testing"
)

func wordPieceMetadata(tokens []string) map[string]MetaValue {
	return map[string]MetaValue{
		"tokenizer.ggml.model":                    {Kind: "str", Value: "bert"},
		"tokenizer.ggml.tokens":                   {Kind: "arr", Value: tokens},
		"tokenizer.ggml.bos_token_id":             {Kind: "u32", Value: uint32(2)},
		"tokenizer.ggml.eos_token_id":             {Kind: "u32", Value: uint32(3)},
		"tokenizer.ggml.unknown_token_id":         {Kind: "u32", Value: uint32(1)},
		"tokenizer.ggml.seperator_token_id":       {Kind: "u32", Value: uint32(3)},
		"tokenizer.ggml.add_bos_token":            {Kind: "bool", Value: true},
		"tokenizer.ggml.add_sep_token":            {Kind: "bool", Value: true},
		"tokenizer.ggml.normalizer.lowercase":     {Kind: "bool", Value: true},
		"tokenizer.ggml.normalizer.strip_accents": {Kind: "bool", Value: true},
	}
}

func TestWordPieceRawVocabulary(t *testing.T) {
	vocab := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "##s", "!"}
	tok, err := TokenizerFromMetadata(wordPieceMetadata(vocab))
	if err != nil {
		t.Fatalf("TokenizerFromMetadata: %v", err)
	}
	if tok.Mode != TokenizerWordPiece || !tok.RawWordPieceVocab {
		t.Fatalf("mode/raw = %v/%v, want WordPiece/raw", tok.Mode, tok.RawWordPieceVocab)
	}

	got := tok.Encode("Hello worlds!")
	want := []uint32{2, 4, 5, 6, 7, 3}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Encode = %v, want %v", got, want)
	}
	if got := tok.DecodeToken(4); got != " hello" {
		t.Fatalf("DecodeToken(first piece) = %q, want leading word boundary", got)
	}
	if got := tok.DecodeToken(6); got != "s" {
		t.Fatalf("DecodeToken(##s) = %q, want s", got)
	}
	var decoded string
	for _, id := range []uint32{4, 5, 6} {
		decoded += tok.DecodeToken(id)
	}
	if decoded != " hello worlds" {
		t.Fatalf("decoded raw WordPiece sequence = %q, want word boundaries", decoded)
	}
}

func TestWordPiecePhantomSpaceGGUFVocabulary(t *testing.T) {
	// This is the vocabulary layout emitted by llama.cpp's BERT converter:
	// ordinary first pieces have ▁, while raw ## continuations lose the ##.
	vocab := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "▁hello", "▁world", "s", "▁!"}
	tok, err := TokenizerFromMetadata(wordPieceMetadata(vocab))
	if err != nil {
		t.Fatalf("TokenizerFromMetadata: %v", err)
	}
	if tok.Mode != TokenizerWordPiece || tok.RawWordPieceVocab {
		t.Fatalf("mode/raw = %v/%v, want WordPiece/phantom-space", tok.Mode, tok.RawWordPieceVocab)
	}

	got := tok.Encode("Hello worlds!")
	want := []uint32{2, 4, 5, 6, 7, 3}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Encode = %v, want %v", got, want)
	}
}

func TestWordPieceUnknownAndNormalizer(t *testing.T) {
	vocab := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "cafe", "known", "cesky"}
	tok, err := TokenizerFromMetadata(wordPieceMetadata(vocab))
	if err != nil {
		t.Fatalf("TokenizerFromMetadata: %v", err)
	}

	got := tok.Encode("CAFÉ known☃")
	want := []uint32{2, 4, 1, 3}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Encode = %v, want %v", got, want)
	}
	if got := tok.Encode("ČESKÝ"); !reflect.DeepEqual(got, []uint32{2, 6, 3}) {
		t.Fatalf("Encode extended accents = %v, want canonical accent stripping", got)
	}
}

func TestWordPieceCorrectedSeparatorMetadataKey(t *testing.T) {
	vocab := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "[ALT_SEP]"}
	meta := wordPieceMetadata(vocab)
	delete(meta, "tokenizer.ggml.seperator_token_id")
	meta["tokenizer.ggml.separator_token_id"] = MetaValue{Kind: "u32", Value: uint32(5)}

	tok, err := TokenizerFromMetadata(meta)
	if err != nil {
		t.Fatalf("TokenizerFromMetadata: %v", err)
	}
	if got := tok.Encode("hello"); !reflect.DeepEqual(got, []uint32{2, 4, 5}) {
		t.Fatalf("Encode = %v, want corrected separator ID", got)
	}
}

func TestWordPieceNormalizerDefaults(t *testing.T) {
	vocab := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "cafe"}
	meta := wordPieceMetadata(vocab)
	delete(meta, "tokenizer.ggml.normalizer.lowercase")
	delete(meta, "tokenizer.ggml.normalizer.strip_accents")

	tok, err := TokenizerFromMetadata(meta)
	if err != nil {
		t.Fatalf("TokenizerFromMetadata: %v", err)
	}
	if !tok.Lowercase || !tok.StripAccents {
		t.Fatalf("normalizer defaults = lowercase:%v strip_accents:%v, want true/true", tok.Lowercase, tok.StripAccents)
	}
	if got := tok.Encode("CAFÉ"); !reflect.DeepEqual(got, []uint32{2, 4, 3}) {
		t.Fatalf("Encode = %v, want default lowercase/accent stripping", got)
	}
}
