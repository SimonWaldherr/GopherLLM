package rag

import (
	"strings"
	"unicode"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// Chunker splits a Doc into overlapping windows sized to fit a retrieval
// budget. The zero value is ready to use.
type Chunker struct {
	// MaxRunes bounds one chunk's size. Zero means 800. If Tokenizer is set,
	// this and OverlapRunes/MinRunes count model tokens instead of runes.
	MaxRunes int
	// OverlapRunes repeats this many trailing runes (or tokens) of a chunk at
	// the start of the next one, so a sentence spanning a chunk boundary is
	// still findable whole in at least one chunk. Zero means 120.
	OverlapRunes int
	// MinRunes is the smallest a trailing chunk may be before it is merged
	// backward into its predecessor instead of standing alone. Zero means
	// 200; a fragment this small on its own is rarely a useful retrieval
	// unit and mostly just dilutes BM25's document-length statistics.
	MinRunes int
	// Tokenizer, when set, makes MaxRunes/OverlapRunes/MinRunes count model
	// tokens rather than runes — the unit retrieval budgets actually care
	// about, since what ultimately matters is how much of the model's
	// context window a chunk consumes.
	Tokenizer *gopherllm.Tokenizer
	// TitlePrefix prepends "Title" (and any heading text tracked by a future
	// Doc field) before a chunk is handed to an Embedder, then strips it
	// before the chunk is shown in a Hit. It affects only the embedded,
	// model-facing text: BM25 indexes doc.Title alongside every chunk's text
	// regardless of this setting, because title terms are cheap to index and
	// among the highest-precision signals available.
	TitlePrefix bool
}

func (c Chunker) maxSize() int {
	if c.MaxRunes > 0 {
		return c.MaxRunes
	}
	return 800
}

func (c Chunker) overlap() int {
	if c.OverlapRunes > 0 {
		return c.OverlapRunes
	}
	return 120
}

// effectiveOverlap caps the configured overlap at half of MaxRunes. Overlap
// bigger than half a window is a contradiction — it would mean each new
// window repeats more of its predecessor than it adds — and without this cap
// a small custom MaxRunes paired with the 120-rune default overlap silently
// blows the size budget it is supposed to respect.
func (c Chunker) effectiveOverlap() int {
	if o := c.overlap(); o <= c.maxSize()/2 {
		return o
	}
	return c.maxSize() / 2
}

func (c Chunker) minSize() int {
	if c.MinRunes > 0 {
		return c.MinRunes
	}
	return 200
}

// unitLen reports s's length in the Chunker's counting unit: tokens when
// Tokenizer is set, runes otherwise.
func (c Chunker) unitLen(s string) int {
	if c.Tokenizer != nil {
		return len(c.Tokenizer.EncodeWithoutBOS(s))
	}
	return len([]rune(s))
}

// Split chunks doc.Text with a paragraph -> sentence -> whitespace -> hard-cut
// cascade: each fallback is reached only when the one above cannot produce a
// unit that fits MaxRunes. The result is deterministic — the same input
// always produces the same offsets — which is what lets a golden-ranking test
// pin a corpus's expected results as literal data.
func (c Chunker) Split(doc Doc) []Chunk {
	text := doc.Text
	if text == "" {
		return nil
	}
	paragraphs := splitParagraphs(text)
	var windows []span
	for _, p := range paragraphs {
		windows = append(windows, c.packUnit(text, p)...)
	}
	windows = mergeShortTrailing(windows, c.minSize(), c.maxSize(), func(s span) int { return c.unitLen(text[s.start:s.end]) })
	chunks := make([]Chunk, len(windows))
	for i, w := range windows {
		chunks[i] = Chunk{DocID: doc.ID, Index: i, Text: text[w.start:w.end], Start: w.start, End: w.end}
	}
	return chunks
}

type span struct{ start, end int }

// packUnit fits one paragraph's byte range [p.start, p.end) into one or more
// windows no larger than MaxRunes, falling back to sentences, then
// whitespace, then a hard cut, and applying OverlapRunes between the windows
// it produces for a paragraph that itself needed splitting.
func (c Chunker) packUnit(text string, p span) []span {
	whole := text[p.start:p.end]
	if c.unitLen(whole) <= c.maxSize() {
		return []span{p}
	}
	sentences := splitSentences(text, p)
	return c.packPieces(text, sentences)
}

// packPieces greedily accumulates consecutive pieces (sentences, or — via
// recursion — smaller units) into windows up to MaxRunes, splitting any
// single piece that alone exceeds MaxRunes with the next fallback down. It
// implements the overlap between consecutive produced windows.
func (c Chunker) packPieces(text string, pieces []span) []span {
	var out []span
	var cur span
	curLen := 0
	flush := func() {
		if curLen > 0 {
			out = append(out, cur)
		}
		cur, curLen = span{}, 0
	}
	for _, piece := range pieces {
		pieceText := text[piece.start:piece.end]
		pieceLen := c.unitLen(pieceText)
		if pieceLen > c.maxSize() {
			flush()
			// This single piece alone doesn't fit: fall back further (words,
			// then a hard cut) and pack its own sub-pieces directly, without
			// re-entering sentence splitting.
			sub := c.splitOversizedPiece(text, piece)
			out = append(out, c.packPieces(text, sub)...)
			continue
		}
		if curLen > 0 && curLen+pieceLen > c.maxSize() {
			flush()
			// Repeat the tail of the previous window at the start of this
			// one, so content near a chunk boundary is still findable whole.
			if overlapStart := overlapPoint(text, out[len(out)-1], c.effectiveOverlap(), c.unitLen); overlapStart < out[len(out)-1].end {
				cur = span{start: overlapStart, end: piece.end}
				curLen = c.unitLen(text[cur.start:cur.end])
				continue
			}
		}
		if curLen == 0 {
			cur = piece
		} else {
			cur.end = piece.end
		}
		curLen += pieceLen
	}
	flush()
	return out
}

// splitOversizedPiece falls back from a sentence that alone exceeds MaxRunes
// to whitespace-separated words, and — if even one word does not fit — a hard
// rune cut. Each returned span already fits MaxRunes on its own.
func (c Chunker) splitOversizedPiece(text string, p span) []span {
	words := splitWhitespace(text, p)
	var out []span
	for _, w := range words {
		if c.unitLen(text[w.start:w.end]) <= c.maxSize() {
			out = append(out, w)
			continue
		}
		out = append(out, c.hardCut(text, w)...)
	}
	if len(out) == 0 {
		return c.hardCut(text, p)
	}
	return out
}

// hardCut splits p into rune-boundary-safe windows of at most MaxRunes each,
// the cascade's last resort for a single unbroken run with no whitespace at
// all (a URL, a hash, a run of CJK/Han text with no ASCII spaces).
func (c Chunker) hardCut(text string, p span) []span {
	var out []span
	runes := []rune(text[p.start:p.end])
	// Map rune index back to a byte offset within p, since Chunk.Start/End
	// are byte offsets into the original Doc.Text.
	byteOffsets := make([]int, len(runes)+1)
	off := p.start
	for i, r := range runes {
		byteOffsets[i] = off
		off += len(string(r))
	}
	byteOffsets[len(runes)] = p.end
	max := c.maxSize()
	if max <= 0 {
		max = 1
	}
	for start := 0; start < len(runes); start += max {
		end := min(start+max, len(runes))
		out = append(out, span{start: byteOffsets[start], end: byteOffsets[end]})
	}
	return out
}

// overlapPoint finds a byte offset within prev such that the unit length from
// that offset to prev.end is approximately overlapUnits, so the next window
// can start there and repeat that tail. Walks backward from prev.end word by
// word (falling back to prev.start if the whole span is shorter than the
// requested overlap) rather than cutting at an arbitrary rune, so an overlap
// never starts mid-word.
func overlapPoint(text string, prev span, overlapUnits int, unitLen func(string) int) int {
	if overlapUnits <= 0 {
		return prev.end
	}
	words := splitWhitespace(text, prev)
	if len(words) == 0 {
		return prev.start
	}
	start := prev.end
	for i := len(words) - 1; i >= 0; i-- {
		candidate := words[i].start
		if unitLen(text[candidate:prev.end]) > overlapUnits {
			break
		}
		start = candidate
	}
	return start
}

// mergeShortTrailing merges the last window into its predecessor when it
// falls below minSize, so a paragraph boundary or an overlap computation
// never leaves a near-empty final chunk standing alone. Only the trailing
// window of the WHOLE document is checked, matching the common failure case
// (a short closing paragraph) without reshuffling interior chunk boundaries a
// caller may already be citing by index.
//
// The merge is skipped when the combined window would exceed maxSize.
// MaxRunes is a hard budget other code (an Embedder's own context limit, a
// caller packing chunks into a model prompt) relies on; a small trailing
// fragment is a retrieval-quality nuisance, but silently violating the size
// contract a Chunker was explicitly configured with is worse.
func mergeShortTrailing(windows []span, minSize, maxSize int, unitLen func(span) int) []span {
	if len(windows) < 2 {
		return windows
	}
	last := windows[len(windows)-1]
	if unitLen(last) >= minSize {
		return windows
	}
	prev := windows[len(windows)-2]
	combined := span{start: prev.start, end: last.end}
	if unitLen(combined) > maxSize {
		return windows
	}
	merged := append([]span(nil), windows[:len(windows)-2]...)
	merged = append(merged, combined)
	return merged
}

// splitParagraphs splits text at one-or-more blank lines. A document with no
// blank line at all is one paragraph spanning the whole text.
func splitParagraphs(text string) []span {
	var out []span
	start := 0
	i := 0
	for i < len(text) {
		if text[i] == '\n' {
			j := i
			for j < len(text) && (text[j] == '\n' || text[j] == '\r') {
				j++
			}
			if strings.Count(text[i:j], "\n") >= 2 {
				if trimmed := trimSpan(text, span{start, i}); trimmed.end > trimmed.start {
					out = append(out, trimmed)
				}
				start = j
				i = j
				continue
			}
			i = j
			continue
		}
		i++
	}
	if trimmed := trimSpan(text, span{start, len(text)}); trimmed.end > trimmed.start {
		out = append(out, trimmed)
	}
	return out
}

// sentenceEnders is deliberately small and un-clever: a period, question
// mark, or exclamation point followed by whitespace (or end of text) ends a
// sentence. This misses abbreviations ("Dr. Smith") and is fine — the
// consequence of a missed boundary is a slightly larger sentence unit, which
// packPieces still fits within MaxRunes; it is not a correctness issue for
// retrieval the way it would be for, say, a text-to-speech pipeline.
func splitSentences(text string, p span) []span {
	var out []span
	start := p.start
	for i := p.start; i < p.end; i++ {
		switch text[i] {
		case '.', '?', '!':
			j := i + 1
			for j < p.end && isSpace(text[j]) {
				j++
			}
			if j > i+1 || j == p.end {
				out = append(out, span{start, j})
				start = j
				i = j - 1
			}
		}
	}
	if start < p.end {
		out = append(out, span{start, p.end})
	}
	if len(out) == 0 {
		return []span{p}
	}
	return out
}

func splitWhitespace(text string, p span) []span {
	var out []span
	start := -1
	for i := p.start; i < p.end; i++ {
		if isSpace(text[i]) {
			if start >= 0 {
				out = append(out, span{start, i})
				start = -1
			}
			continue
		}
		if start < 0 {
			start = i
		}
	}
	if start >= 0 {
		out = append(out, span{start, p.end})
	}
	return out
}

func isSpace(b byte) bool { return b == ' ' || b == '\t' || b == '\n' || b == '\r' }

func trimSpan(text string, s span) span {
	for s.start < s.end && unicode.IsSpace(rune(text[s.start])) {
		s.start++
	}
	for s.end > s.start && unicode.IsSpace(rune(text[s.end-1])) {
		s.end--
	}
	return s
}
