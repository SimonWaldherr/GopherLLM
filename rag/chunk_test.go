package rag

import (
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func TestChunkerShortDocIsOneChunk(t *testing.T) {
	c := Chunker{}
	chunks := c.Split(Doc{ID: "d1", Text: "A short document that easily fits in one chunk."})
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want 1: %+v", len(chunks), chunks)
	}
	if chunks[0].Text != "A short document that easily fits in one chunk." {
		t.Fatalf("chunk text = %q", chunks[0].Text)
	}
	if chunks[0].Start != 0 || chunks[0].End != len(chunks[0].Text) {
		t.Fatalf("offsets = %d,%d", chunks[0].Start, chunks[0].End)
	}
}

func TestChunkerEmptyDocProducesNoChunks(t *testing.T) {
	if got := (Chunker{}).Split(Doc{ID: "d1", Text: ""}); got != nil {
		t.Fatalf("got %v, want nil", got)
	}
}

func TestChunkerCascadeFallsBackInOrder(t *testing.T) {
	// A paragraph that fits: one chunk.
	c := Chunker{MaxRunes: 50}
	short := c.Split(Doc{ID: "d", Text: "One short paragraph."})
	if len(short) != 1 {
		t.Fatalf("short: got %d chunks, want 1", len(short))
	}

	// A paragraph too long to fit whole, but its sentences do: falls back to
	// sentence splitting.
	sentenceLevel := c.Split(Doc{ID: "d", Text: strings.Repeat("Word word word. ", 10)})
	if len(sentenceLevel) < 2 {
		t.Fatalf("expected multiple chunks from sentence splitting, got %d", len(sentenceLevel))
	}
	for _, ch := range sentenceLevel {
		if len([]rune(ch.Text)) > c.MaxRunes {
			t.Fatalf("chunk exceeds budget: %d runes: %q", len([]rune(ch.Text)), ch.Text)
		}
	}

	// A single "sentence" with no punctuation, longer than MaxRunes: falls
	// back to whitespace splitting.
	wordLevel := c.Split(Doc{ID: "d", Text: strings.Repeat("wordy ", 20)})
	if len(wordLevel) < 2 {
		t.Fatalf("expected multiple chunks from word splitting, got %d", len(wordLevel))
	}

	// A single unbroken run with no whitespace at all: falls back to a hard
	// rune cut.
	hardCut := c.Split(Doc{ID: "d", Text: strings.Repeat("x", 200)})
	if len(hardCut) < 2 {
		t.Fatalf("expected multiple chunks from a hard cut, got %d", len(hardCut))
	}
	for _, ch := range hardCut {
		if len([]rune(ch.Text)) > c.MaxRunes {
			t.Fatalf("hard-cut chunk exceeds MaxRunes: %d", len([]rune(ch.Text)))
		}
	}
}

func TestChunkerOverlapRepeatsTail(t *testing.T) {
	c := Chunker{MaxRunes: 32, OverlapRunes: 10}
	text := strings.Repeat("alpha beta gamma delta. ", 6)
	chunks := c.Split(Doc{ID: "d", Text: text})
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
	for i := 1; i < len(chunks); i++ {
		if chunks[i].Start >= chunks[i-1].End {
			t.Fatalf("chunk %d does not overlap chunk %d: [%d,%d) vs [%d,%d)",
				i, i-1, chunks[i].Start, chunks[i].End, chunks[i-1].Start, chunks[i-1].End)
		}
	}
}

func TestChunkerDropsOverlapThatWouldExceedBudget(t *testing.T) {
	// Each sentence fits, but even its shortest whole-word overlap would
	// push the following sentence beyond the hard 30-rune budget.
	c := Chunker{MaxRunes: 30, OverlapRunes: 10}
	text := strings.Repeat("alpha beta gamma delta. ", 6)
	chunks := c.Split(Doc{ID: "d", Text: text})
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
	for _, chunk := range chunks {
		if got := c.unitLen(chunk.Text); got > c.MaxRunes {
			t.Fatalf("overlap exceeded budget: %d > %d: %q", got, c.MaxRunes, chunk.Text)
		}
		if text[chunk.Start:chunk.End] != chunk.Text {
			t.Fatal("overlap changed the source offsets")
		}
	}
}

func TestChunkerCountsJoinedWhitespaceInTokenBudget(t *testing.T) {
	// With no BPE merges, ASCII letters and the separating space each cost
	// one token. Summing the individual two-letter words misses that space.
	tok := &gopherllm.Tokenizer{
		Mode: gopherllm.TokenizerGPT2BPE, TokenToID: make(map[string]uint32),
		ByteEncoder: make(map[byte]rune),
	}
	for _, r := range "abcdefghijklmnopqrstuvwxyz " {
		tok.TokenToID[string(r)] = uint32(len(tok.Vocab))
		tok.Vocab = append(tok.Vocab, string(r))
		tok.ByteEncoder[byte(r)] = r
	}
	c := Chunker{MaxRunes: 4, Tokenizer: tok}
	chunks := c.Split(Doc{ID: "d", Text: "ab cd ef gh ij"})
	if len(chunks) != 5 {
		t.Fatalf("expected five separate words, got %+v", chunks)
	}
	for _, chunk := range chunks {
		if got := c.unitLen(chunk.Text); got > c.MaxRunes {
			t.Fatalf("joined text exceeds token budget: %d > %d: %q", got, c.MaxRunes, chunk.Text)
		}
	}
}

func byteFallbackChunkTokenizer() *gopherllm.Tokenizer {
	tok := &gopherllm.Tokenizer{
		Mode: gopherllm.TokenizerSentencePiece, Vocab: []string{"▁"},
		TokenToID: map[string]uint32{"▁": 0},
	}
	for b := 0; b < 256; b++ {
		name := fmt.Sprintf("<0x%02X>", b)
		tok.TokenToID[name] = uint32(len(tok.Vocab))
		tok.Vocab = append(tok.Vocab, name)
	}
	return tok
}

func TestChunkerHardCutHonorsByteFallbackTokenBudget(t *testing.T) {
	c := Chunker{MaxRunes: 7, OverlapRunes: 1, Tokenizer: byteFallbackChunkTokenizer()}
	text := strings.Repeat("漢字😀", 9)
	chunks := c.Split(Doc{ID: "d", Text: text})
	if len(chunks) < 2 {
		t.Fatalf("expected token-budget cuts, got %+v", chunks)
	}
	var joined strings.Builder
	for _, chunk := range chunks {
		if !utf8.ValidString(chunk.Text) || chunk.Start >= chunk.End || text[chunk.Start:chunk.End] != chunk.Text {
			t.Fatalf("cut corrupted UTF-8 or source offsets: %+v", chunk)
		}
		if got := c.unitLen(chunk.Text); got > c.MaxRunes {
			t.Fatalf("byte fallback exceeds token budget: %d > %d: %q", got, c.MaxRunes, chunk.Text)
		}
		joined.WriteString(chunk.Text)
	}
	if joined.String() != text {
		t.Fatal("hard cuts omitted or repeated input runes")
	}
}

func TestChunkerUnrepresentableRuneStillMakesProgress(t *testing.T) {
	c := Chunker{MaxRunes: 1, Tokenizer: byteFallbackChunkTokenizer()}
	text := "漢😀字"
	chunks := c.Split(Doc{ID: "d", Text: text})
	if len(chunks) != utf8.RuneCountInString(text) {
		t.Fatalf("expected one indivisible rune per chunk, got %+v", chunks)
	}
	var joined strings.Builder
	for _, chunk := range chunks {
		if !utf8.ValidString(chunk.Text) || utf8.RuneCountInString(chunk.Text) != 1 || text[chunk.Start:chunk.End] != chunk.Text {
			t.Fatalf("unsplittable rune was corrupted: %+v", chunk)
		}
		joined.WriteString(chunk.Text)
	}
	if joined.String() != text {
		t.Fatal("unsplittable rune was lost")
	}
}

func TestChunkerMergesTrailingFragment(t *testing.T) {
	c := Chunker{MaxRunes: 40, OverlapRunes: 0, MinRunes: 15}
	// Built so the final paragraph is short enough to trigger the merge.
	text := strings.Repeat("A reasonably sized paragraph of text. ", 4) + "\n\n" + "Tiny."
	chunks := c.Split(Doc{ID: "d", Text: text})
	if len(chunks) == 0 {
		t.Fatal("no chunks produced")
	}
	last := chunks[len(chunks)-1]
	if !strings.HasSuffix(last.Text, "Tiny.") {
		t.Fatalf("last chunk = %q, want it to end with the merged trailing fragment", last.Text)
	}
}

func TestChunkerOffsetsRoundTripIntoSourceText(t *testing.T) {
	c := Chunker{MaxRunes: 25, OverlapRunes: 5}
	text := "The quick brown fox jumps over the lazy dog. It was a sunny afternoon in the park."
	chunks := c.Split(Doc{ID: "d", Text: text})
	for _, ch := range chunks {
		if text[ch.Start:ch.End] != ch.Text {
			t.Fatalf("chunk offsets do not round-trip: [%d,%d) = %q, chunk.Text = %q", ch.Start, ch.End, text[ch.Start:ch.End], ch.Text)
		}
	}
}

func TestChunkerTokenizerBudgetCountsTokensNotRunes(t *testing.T) {
	// Without a Tokenizer, MaxRunes counts runes directly.
	plain := Chunker{MaxRunes: 5}
	chunks := plain.Split(Doc{ID: "d", Text: "abcdefghij"})
	if len(chunks) < 2 {
		t.Fatalf("expected a rune-budget split, got %d chunks", len(chunks))
	}
}

func TestChunkerDeterministicAcrossRuns(t *testing.T) {
	c := Chunker{MaxRunes: 30, OverlapRunes: 8}
	text := strings.Repeat("Repeatable input text for determinism. ", 8)
	a := c.Split(Doc{ID: "d", Text: text})
	b := c.Split(Doc{ID: "d", Text: text})
	if len(a) != len(b) {
		t.Fatalf("lengths differ: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("chunk %d differs: %+v vs %+v", i, a[i], b[i])
		}
	}
}
