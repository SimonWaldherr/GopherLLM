package gopherllm

import (
	"math"
	"math/rand"
	"strings"
	"testing"
)

// The reference implementations below are the linear-scan merge loops the
// heap replaced. They are kept here so the fast paths stay bit-identical:
// every merge decision, including the leftmost-wins tie-break, must match.

func (t *Tokenizer) referenceSentencePiece(text string) []uint32 {
	processed := strings.ReplaceAll(" "+text, " ", "\u2581")
	current := t.encodeFromPieces(strings.Split(processed, ""))
	for len(current) >= 2 {
		bestScore := float32(-3.4e38)
		bestIdx := -1
		var bestID uint32
		for i := 0; i+1 < len(current); i++ {
			merged := t.decodeRaw(current[i]) + t.decodeRaw(current[i+1])
			if id, ok := t.TokenToID[merged]; ok {
				score := float32(0)
				if int(id) < len(t.Scores) {
					score = t.Scores[id]
				}
				if score > bestScore {
					bestScore, bestIdx, bestID = score, i, id
				}
			}
		}
		if bestIdx < 0 {
			break
		}
		current[bestIdx] = bestID
		current = append(current[:bestIdx+1], current[bestIdx+2:]...)
	}
	return current
}

func (t *Tokenizer) referenceGPT2BPE(text string) []uint32 {
	out := []uint32{}
	for _, piece := range t.pretokenize(text) {
		var encoded strings.Builder
		for _, b := range []byte(piece) {
			if ch, ok := t.ByteEncoder[b]; ok {
				encoded.WriteRune(ch)
			}
		}
		symbols := strings.Split(encoded.String(), "")
		for len(symbols) > 1 {
			bestRank := int(^uint(0) >> 1)
			bestIdx := -1
			for i := 0; i+1 < len(symbols); i++ {
				if rank, ok := t.MergeRanks[Pair{symbols[i], symbols[i+1]}]; ok && rank < bestRank {
					bestRank, bestIdx = rank, i
				}
			}
			if bestIdx < 0 {
				break
			}
			symbols[bestIdx] += symbols[bestIdx+1]
			symbols = append(symbols[:bestIdx+1], symbols[bestIdx+2:]...)
		}
		for _, symbol := range symbols {
			if id, ok := t.TokenToID[symbol]; ok {
				out = append(out, id)
			} else {
				out = append(out, t.encodeFromPieces([]string{symbol})...)
			}
		}
	}
	return out
}

// randomSentencePieceVocab builds a vocabulary whose merges are ambiguous on
// purpose: scores repeat, so tie-breaking is exercised, and only some
// substrings exist, so the byte fallback runs too.
func randomSentencePieceVocab(rng *rand.Rand, corpus string, coverage float64, tieBuckets int) *Tokenizer {
	vocab := []string{"<unk>", "<s>", "</s>"}
	seen := map[string]bool{"<unk>": true, "<s>": true, "</s>": true}
	add := func(s string) {
		if s != "" && !seen[s] {
			seen[s] = true
			vocab = append(vocab, s)
		}
	}
	for i := 0; i < 256; i++ {
		add(spByteTokens[i])
	}
	for _, r := range corpus {
		if rng.Float64() < 0.9 {
			add(string(r))
		}
	}
	add("\u2581")
	runes := []rune(strings.ReplaceAll(corpus, " ", "\u2581"))
	for start := 0; start < len(runes); start++ {
		for l := 2; l <= 6 && start+l <= len(runes); l++ {
			if rng.Float64() < coverage {
				add(string(runes[start : start+l]))
			}
		}
	}
	scores := make([]float32, len(vocab))
	toID := make(map[string]uint32, len(vocab))
	for i, v := range vocab {
		toID[v] = uint32(i)
		scores[i] = float32(rng.Intn(tieBuckets))
	}
	return &Tokenizer{Vocab: vocab, Scores: scores, TokenToID: toID,
		Mode: TokenizerSentencePiece, AddBOS: true, BOSID: 1, EOSID: 2}
}

func randomGPT2Vocab(rng *rand.Rand, corpus string, coverage float64, tieBuckets int) *Tokenizer {
	enc, dec := buildByteMaps()
	ranks := map[Pair]int{}
	vocab := []string{"<unk>", "<s>", "</s>"}
	seen := map[string]bool{"<unk>": true, "<s>": true, "</s>": true}
	add := func(s string) {
		if s != "" && !seen[s] {
			seen[s] = true
			vocab = append(vocab, s)
		}
	}
	for i := 0; i < 256; i++ {
		add(spByteTokens[i])
	}
	var mapped strings.Builder
	for i := 0; i < len(corpus); i++ {
		if ch, ok := enc[corpus[i]]; ok {
			mapped.WriteRune(ch)
		}
	}
	runes := []rune(mapped.String())
	for _, r := range runes {
		add(string(r))
	}
	for start := 0; start < len(runes); start++ {
		for l := 2; l <= 6 && start+l <= len(runes); l++ {
			if rng.Float64() >= coverage {
				continue
			}
			piece := string(runes[start : start+l])
			add(piece)
			for split := 1; split < l; split++ {
				ranks[Pair{string(runes[start : start+split]), string(runes[start+split : start+l])}] = rng.Intn(tieBuckets)
			}
		}
	}
	toID := make(map[string]uint32, len(vocab))
	for i, v := range vocab {
		toID[v] = uint32(i)
	}
	return &Tokenizer{Vocab: vocab, TokenToID: toID, MergeRanks: ranks,
		ByteEncoder: enc, ByteDecoder: dec, Mode: TokenizerGPT2BPE}
}

var mergeCorpora = []string{
	"Der schnelle braune Fuchs springt über den faulen Hund.",
	"hello world this is a benchmark of the tokenizer merge loop",
	"Mixed CASE, numbers 2024, punctuation... and $999 prices!",
	"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	"日本語のテキストはスペースで区切られない",
	"emoji 🙂🙃 and combining e\u0301 marks",
	"\t\n  leading and trailing whitespace  \n",
	"",
	"a",
	"ab",
}

// forEachMergePath runs fn under the default threshold and under both
// extremes, so every case is checked against the linear scan on the heap path
// and on the scan path alike.
func forEachMergePath(t *testing.T, fn func(t *testing.T)) {
	t.Helper()
	original := bpeScanMaxSymbols
	t.Cleanup(func() { bpeScanMaxSymbols = original })
	for _, path := range []struct {
		name      string
		threshold int
	}{
		{"default", original},
		{"heap", 0},
		{"scan", math.MaxInt32},
	} {
		bpeScanMaxSymbols = path.threshold
		t.Run(path.name, fn)
	}
}

func TestEncodeSentencePieceMatchesLinearScan(t *testing.T) {
	forEachMergePath(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(7))
		for _, coverage := range []float64{0.15, 0.5, 1.0} {
			for _, ties := range []int{1, 3, 64} {
				for _, corpus := range mergeCorpora {
					tok := randomSentencePieceVocab(rng, corpus, coverage, ties)
					want := tok.referenceSentencePiece(corpus)
					got := tok.encodeSentencePiece(corpus)
					if !equalIDs(got, want) {
						t.Fatalf("sentencepiece coverage=%v ties=%d corpus=%q:\n got %v\nwant %v",
							coverage, ties, corpus, got, want)
					}
				}
			}
		}
	})
}

func TestEncodeGPT2BPEMatchesLinearScan(t *testing.T) {
	forEachMergePath(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(11))
		for _, pre := range []string{"", "qwen35", "tekken", "kimi"} {
			for _, coverage := range []float64{0.15, 0.5, 1.0} {
				for _, ties := range []int{1, 3, 64} {
					for _, corpus := range mergeCorpora {
						tok := randomGPT2Vocab(rng, corpus, coverage, ties)
						tok.Pre = pre
						want := tok.referenceGPT2BPE(corpus)
						got := tok.encodeGPT2BPE(corpus)
						if !equalIDs(got, want) {
							t.Fatalf("gpt2 pre=%q coverage=%v ties=%d corpus=%q:\n got %v\nwant %v",
								pre, coverage, ties, corpus, got, want)
						}
					}
				}
			}
		}
	})
}

// TestEncodeMergeRandomTextMatchesLinearScan fuzzes the character mix rather
// than the vocabulary, covering byte fallback, unmergeable runs, and symbol
// counts on both sides of the scan/heap threshold.
func TestEncodeMergeRandomTextMatchesLinearScan(t *testing.T) {
	forEachMergePath(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(23))
		alphabet := []rune("abc ▁·äß日🙂\n\t.,!0123")
		for iter := 0; iter < 200; iter++ {
			n := rng.Intn(120)
			var sb strings.Builder
			for i := 0; i < n; i++ {
				sb.WriteRune(alphabet[rng.Intn(len(alphabet))])
			}
			text := sb.String()
			sp := randomSentencePieceVocab(rng, text, rng.Float64(), 1+rng.Intn(8))
			if got, want := sp.encodeSentencePiece(text), sp.referenceSentencePiece(text); !equalIDs(got, want) {
				t.Fatalf("sentencepiece iter=%d text=%q:\n got %v\nwant %v", iter, text, got, want)
			}
			gp := randomGPT2Vocab(rng, text, rng.Float64(), 1+rng.Intn(8))
			if got, want := gp.encodeGPT2BPE(text), gp.referenceGPT2BPE(text); !equalIDs(got, want) {
				t.Fatalf("gpt2 iter=%d text=%q:\n got %v\nwant %v", iter, text, got, want)
			}
		}
	})
}

func equalIDs(a, b []uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
