package tokenizer

import (
	"math/rand"
	"reflect"
	"strings"
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

// Long prompts and unspaced scripts are what the merge heap exists for: both
// used to be quadratic in the symbol run, so a chat turn paid seconds of
// tokenizer time before the first token was even embedded.

const benchLongPrompt = "Der schnelle braune Fuchs springt ueber den faulen Hund und fragt sich warum die Sonne scheint. "

func BenchmarkEncodeSentencePieceLongPrompt(b *testing.B) {
	corpus := strings.Repeat(benchLongPrompt, 44)[:4096]
	tok := randomSentencePieceVocab(rand.New(rand.NewSource(1)), corpus, 0.6, 64)
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.encodeSentencePiece(corpus)
	}
}

func BenchmarkEncodeGPT2BPEWords(b *testing.B) {
	corpus := strings.Repeat(benchLongPrompt, 11)
	tok := randomGPT2Vocab(rand.New(rand.NewSource(1)), corpus, 0.6, 4096)
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.encodeGPT2BPE(corpus)
	}
}

// The GPT-2 pretokenizers emit \p{L}+ runs, so a Japanese sentence arrives at
// the merge loop as one symbol run of unbounded length.
func BenchmarkEncodeGPT2BPEUnspaced(b *testing.B) {
	corpus := strings.Repeat("日本語のテキストはスペースで区切られないので長い語になる", 25)
	tok := randomGPT2Vocab(rand.New(rand.NewSource(1)), corpus, 0.6, 4096)
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.encodeGPT2BPE(corpus)
	}
}
