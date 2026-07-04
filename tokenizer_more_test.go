package main

import (
	"reflect"
	"testing"
)

func TestEncodeSentencePieceMergesByScore(t *testing.T) {
	vocab := []string{"<unk>", "<s>", "</s>", "▁", "h", "i", "hi", "▁hi"}
	scores := []float32{0, 0, 0, 0, 0, 0, 1, 2}
	toID := make(map[string]uint32, len(vocab))
	for i, v := range vocab {
		toID[v] = uint32(i)
	}
	tok := &Tokenizer{Vocab: vocab, Scores: scores, TokenToID: toID, Mode: TokenizerSentencePiece, AddBOS: true, BOSID: 1, EOSID: 2}

	got := tok.encodeSentencePiece("hi")
	if want := []uint32{toID["▁hi"]}; !reflect.DeepEqual(got, want) {
		t.Fatalf("encodeSentencePiece(hi) = %v, want %v", got, want)
	}
	// Encode prepends BOS.
	if full := tok.Encode("hi"); len(full) != 2 || full[0] != tok.BOSID || full[1] != toID["▁hi"] {
		t.Fatalf("Encode(hi) = %v", full)
	}
	// DecodeToken maps the ▁ marker back to a leading space.
	if s := tok.DecodeToken(toID["▁hi"]); s != " hi" {
		t.Fatalf("DecodeToken = %q, want %q", s, " hi")
	}
}

func TestSentencePieceByteFallback(t *testing.T) {
	vocab := []string{"<unk>", "<s>", "</s>", "▁", "<0x41>", "<0x42>"}
	toID := make(map[string]uint32, len(vocab))
	for i, v := range vocab {
		toID[v] = uint32(i)
	}
	tok := &Tokenizer{Vocab: vocab, Scores: make([]float32, len(vocab)), TokenToID: toID, Mode: TokenizerSentencePiece, BOSID: 1, EOSID: 2}
	// 'A' (0x41) and 'B' (0x42) are not in the vocab as characters, so they fall
	// back to byte tokens.
	got := tok.encodeFromPieces([]string{"A", "B"})
	if want := []uint32{toID["<0x41>"], toID["<0x42>"]}; !reflect.DeepEqual(got, want) {
		t.Fatalf("byte fallback = %v, want %v", got, want)
	}
	// And those byte tokens decode back to the raw bytes.
	if s := tok.DecodeToken(toID["<0x41>"]); s != "A" {
		t.Fatalf("DecodeToken(<0x41>) = %q, want A", s)
	}
}

func TestBuildByteMapsAreInverse(t *testing.T) {
	enc, dec := buildByteMaps()
	if len(enc) != 256 {
		t.Fatalf("byte encoder has %d entries, want 256", len(enc))
	}
	for b := 0; b < 256; b++ {
		r, ok := enc[byte(b)]
		if !ok {
			t.Fatalf("byte %d not encoded", b)
		}
		if dec[r] != byte(b) {
			t.Fatalf("decoder(%q) = %d, want %d", r, dec[r], b)
		}
	}
}

func TestGPT2DecodeRoundTrip(t *testing.T) {
	enc, dec := buildByteMaps()
	tok := &Tokenizer{ByteEncoder: enc, ByteDecoder: dec, Mode: TokenizerGPT2BPE}
	// Encode the bytes of " hi" through the byte encoder, then decode back.
	var encoded []rune
	for _, b := range []byte(" hi") {
		encoded = append(encoded, enc[b])
	}
	if got := tok.decodeGPT2Bytes(string(encoded)); got != " hi" {
		t.Fatalf("decodeGPT2Bytes = %q, want %q", got, " hi")
	}
}

func TestEncodeWithoutBOSEmpty(t *testing.T) {
	tok := newInstTestTokenizer()
	if got := tok.EncodeWithoutBOS(""); got != nil {
		t.Fatalf("empty encode = %v, want nil", got)
	}
}

func TestSpecialIDLookup(t *testing.T) {
	tok := newInstTestTokenizer()
	if id, ok := tok.SpecialID("[INST]"); !ok || id != tok.TokenToID["[INST]"] {
		t.Fatalf("SpecialID([INST]) = %d,%v", id, ok)
	}
	if _, ok := tok.SpecialID("[NOPE]"); ok {
		t.Fatal("SpecialID for missing token should be false")
	}
}
