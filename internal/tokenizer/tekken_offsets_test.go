package tokenizer

import (
	"crypto/sha256"
	"math/rand"
	"os"
	"slices"

	gguf "github.com/SimonWaldherr/GopherLLM/internal/formats/gguf"
	"github.com/SimonWaldherr/GopherLLM/internal/mmapfile"
	"reflect"
	"strings"
	"testing"
	"unicode"
)

func referencePretokenizeTekken(text string) []string {
	r := []rune(text)
	n := len(r)
	pieces := make([]string, 0, len(r)/3+1)
	i := 0
	for i < n {
		if end, ok := referenceMatchTekkenWord(r, n, i); ok {
			pieces = append(pieces, string(r[i:end]))
			i = end
			continue
		}
		if referenceIsTekkenNumber(r[i]) {
			pieces = append(pieces, string(r[i:i+1]))
			i++
			continue
		}
		if end, ok := referenceMatchTekkenPunct(r, n, i); ok {
			pieces = append(pieces, string(r[i:end]))
			i = end
			continue
		}
		if unicode.IsSpace(r[i]) {
			end := referenceTekkenWhitespaceEnd(r, n, i)
			pieces = append(pieces, string(r[i:end]))
			i = end
			continue
		}
		pieces = append(pieces, string(r[i:i+1]))
		i++
	}
	return pieces
}

func referenceIsTekkenNumber(c rune) bool { return unicode.IsNumber(c) }

// referenceTekkenUpperClass matches [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}].
func referenceTekkenUpperClass(c rune) bool {
	return (unicode.IsLetter(c) && !unicode.IsLower(c)) || unicode.Is(unicode.M, c)
}

// referenceTekkenLowerClass matches [\p{Ll}\p{Lm}\p{Lo}\p{M}].
func referenceTekkenLowerClass(c rune) bool {
	return (unicode.IsLetter(c) && !unicode.IsUpper(c) && !unicode.IsTitle(c)) || unicode.Is(unicode.M, c)
}

// referenceMatchTekkenWord handles the two letter alternatives (optional leading
// non-letter/digit character, then case-split letter runs).
func referenceMatchTekkenWord(r []rune, n, i int) (int, bool) {
	opt := i
	if opt < n && r[opt] != '\r' && r[opt] != '\n' && !unicode.IsLetter(r[opt]) && !referenceIsTekkenNumber(r[opt]) {
		opt++
	}
	// Alt 1: U* L+
	k := opt
	for k < n && referenceTekkenUpperClass(r[k]) {
		k++
	}
	lStart := k
	for k < n && referenceTekkenLowerClass(r[k]) {
		k++
	}
	if k > lStart {
		return k, true
	}
	// Alt 2: U+ L*
	k = opt
	for k < n && referenceTekkenUpperClass(r[k]) {
		k++
	}
	if k > opt {
		for k < n && referenceTekkenLowerClass(r[k]) {
			k++
		}
		return k, true
	}
	return i, false
}

// referenceMatchTekkenPunct handles " ?[^\s\p{L}\p{N}]+[\r\n/]*".
func referenceMatchTekkenPunct(r []rune, n, i int) (int, bool) {
	j := i
	if j < n && r[j] == ' ' {
		j++
	}
	pStart := j
	for j < n && !unicode.IsSpace(r[j]) && !unicode.IsLetter(r[j]) && !referenceIsTekkenNumber(r[j]) {
		j++
	}
	if j == pStart {
		return i, false
	}
	for j < n && (r[j] == '\r' || r[j] == '\n' || r[j] == '/') {
		j++
	}
	return j, true
}

// referenceTekkenWhitespaceEnd handles "\s*[\r\n]+ | \s+(?!\S) | \s+": a whitespace run
// that ends at its final newline is cut there, otherwise the whole run is one
// piece.
func referenceTekkenWhitespaceEnd(r []rune, n, i int) int {
	k := i
	lastNL := -1
	for k < n && unicode.IsSpace(r[k]) {
		if r[k] == '\r' || r[k] == '\n' {
			lastNL = k
		}
		k++
	}
	if lastNL >= 0 {
		return lastNL + 1
	}
	return k
}

func TestTekkenOffsetsMatchRuneSplitter(t *testing.T) {
	cases := []string{
		"", " ", "\r\n \t ", "HTTPServer2 XMLHttpRequest １２٣½", "HelloWORLD testASCII",
		"Grüße aus Köln! L'été à Paris; España, România. Ελληνικά Кириллица.",
		"a\u0301 \u0301A ǅelta ᵃʰᴬ 中文かな العربية 日本語",
		"[INST] Réponds en français. [/INST] <|audio|>🙂✨\r\n  /",
		"\x00\xff\xfe\xc0\xaf\xed\xa0\x80\xf0\x80\x80\x80\xe2\x82",
	}
	check := func(text string) {
		t.Helper()
		got, want := pretokenizeTekken(text), referencePretokenizeTekken(text)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("split %q:\n got %q\nwant %q", text, got, want)
		}
	}
	for _, text := range cases {
		check(text)
	}
	rng := rand.New(rand.NewSource(82026))
	alphabet := []rune("aAzZǅßᵃᴬ\u0301\u200d日١²½🙂\r\n\t\v\f \u00a0\u2028/[]!0")
	for iter := 0; iter < 3000; iter++ {
		runes := make([]rune, rng.Intn(160))
		for i := range runes {
			if rng.Intn(5) == 0 {
				runes[i] = rune(rng.Intn(unicode.MaxRune + 1))
			} else {
				runes[i] = alphabet[rng.Intn(len(alphabet))]
			}
		}
		check(string(runes))
		bytes := make([]byte, rng.Intn(100))
		rng.Read(bytes)
		check(string(bytes))
	}
}

func FuzzTekkenOffsets(f *testing.F) {
	for _, text := range []string{"", "CamelCASE １２34", "e\u0301\r\n!", "\xff\xff"} {
		f.Add(text)
	}
	f.Fuzz(func(t *testing.T, text string) {
		if got, want := pretokenizeTekken(text), referencePretokenizeTekken(text); !reflect.DeepEqual(got, want) {
			t.Fatalf("split %q:\n got %q\nwant %q", text, got, want)
		}
	})
}

func TestEncodeTekkenOffsetsMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	for _, text := range append(append([]string(nil), mergeCorpora...), "[INST] Grüße à Paris! ١٢34 [/INST]", "a\xff\xfe\x80日") {
		for _, coverage := range []float64{0.1, 0.6, 1.0} {
			tok := randomGPT2Vocab(rng, text, coverage, 16)
			tok.Pre = "tekken"
			if got, want := tok.encodeGPT2BPE(text), referenceEncodeTekken(tok, text); !equalIDs(got, want) {
				t.Fatalf("encode %q:\n got %v\nwant %v", text, got, want)
			}
		}
	}
}

// The original encode loop isolates splitter costs while retaining exactly the
// same vocabulary and merge algorithm in full-tokenization measurements.
func referenceEncodeTekken(t *Tokenizer, text string) []uint32 {
	return referenceEncodeGPT2Pieces(t, text, referencePretokenizeTekken(text))
}

func referenceEncodeGPT2PieceBuffers(t *Tokenizer, text string) []uint32 {
	return referenceEncodeGPT2Pieces(t, text, t.pretokenize(text))
}

func referenceEncodeGPT2Pieces(t *Tokenizer, text string, pieces []string) []uint32 {
	out := make([]uint32, 0, len(text)/3+8)
	var syms []bpeSymbol
	var heap bpeHeap
	for _, piece := range pieces {
		var encoded strings.Builder
		encoded.Grow(2 * len(piece))
		for i := 0; i < len(piece); i++ {
			if ch, ok := t.ByteEncoder[piece[i]]; ok {
				encoded.WriteRune(ch)
			}
		}
		mapped := encoded.String()
		if mapped == "" {
			continue
		}
		syms = bpeSymbolsFromRunes(syms, mapped)
		mergeBPESymbols(syms, &heap,
			func(l, r *bpeSymbol) (float64, uint32, bool) {
				rank, ok := t.MergeRanks[Pair{l.text, r.text}]
				return float64(rank), 0, ok
			},
			func(l, r *bpeSymbol, _ uint32) string { return l.text + r.text })
		for i := int32(0); i >= 0; i = syms[i].next {
			out = t.appendPieceTokens(out, syms[i].text)
		}
	}
	return out
}

var tekkenPiecesSink []string
var tekkenIDsSink []uint32

func tekkenUniqueCorpus() string {
	var out strings.Builder
	for i := 0; i < 1024; i++ {
		// Unique words exercise allocation reuse without repeated-piece caching.
		out.WriteString(" Wort")
		for shift := 0; shift < 4; shift++ {
			out.WriteByte('a' + byte(i>>(shift*3)&7))
		}
	}
	return out.String()
}

func BenchmarkTekkenOffsets(b *testing.B) {
	var checkpoint *Tokenizer
	if path := os.Getenv("GOPHERLLM_MINISTRAL_TOKENIZER_MODEL"); path != "" {
		mapped, err := mmapfile.Open(path)
		if err != nil {
			b.Fatal(err)
		}
		b.Cleanup(func() { mapped.Close() })
		parsed, err := gguf.ParseQuiet(mapped.Bytes())
		if err != nil {
			b.Fatal(err)
		}
		b.Logf("checkpoint_header_sha256=%x", sha256.Sum256(mapped.Bytes()[:parsed.DataOffset]))
		checkpoint, err = TokenizerFromMetadata(parsed.Metadata)
		if err != nil {
			b.Fatal(err)
		}
		if checkpoint.Mode != TokenizerGPT2BPE || !strings.Contains(checkpoint.Pre, "tekken") {
			b.Fatalf("checkpoint does not use Tekken: %v/%q", checkpoint.Mode, checkpoint.Pre)
		}
	}

	for _, corpus := range []struct{ name, text string }{
		{"short", "Erkläre mir bitte HTTPServer2."},
		{"ascii", strings.Repeat("Explain how HTTPServer2 runs a program. ", 32)},
		{"unique", tekkenUniqueCorpus()},
		{"long_unique", strings.Repeat(tekkenUniqueCorpus(), 8)},
		{"long_piece", "prefix " + strings.Repeat("日本語", 8192)},
		{"european", strings.Repeat("Grüße aus Köln! L'été à Paris; España, România. Ελληνικά Кириллица.\n", 32)},
		{"digits", strings.Repeat("1234567890 １２٣½\r\n", 64)},
		{"unspaced", strings.Repeat("日本語のテキストはスペースで区切られない", 32)},
	} {
		b.Run(corpus.name, func(b *testing.B) {
			splitters := []struct {
				name  string
				split func(string) []string
			}{
				{"reference", referencePretokenizeTekken},
				{"offsets", pretokenizeTekken},
			}
			if os.Getenv("TEKKEN_BENCH_REVERSE") == "1" {
				slices.Reverse(splitters)
			}
			for _, splitter := range splitters {
				b.Run("split/"+splitter.name, func(b *testing.B) {
					b.SetBytes(int64(len(corpus.text)))
					b.ReportAllocs()
					for b.Loop() {
						tekkenPiecesSink = splitter.split(corpus.text)
					}
				})
			}
			tok := checkpoint
			if tok == nil {
				tok = randomGPT2Vocab(rand.New(rand.NewSource(1)), corpus.text, 0.6, 4096)
				tok.Pre = "tekken"
			}
			want := referenceEncodeTekken(tok, corpus.text)
			if !equalIDs(tok.encodeGPT2BPE(corpus.text), want) || !equalIDs(referenceEncodeGPT2PieceBuffers(tok, corpus.text), want) {
				b.Fatal("checkpoint token IDs differ from reference")
			}
			encoders := []struct {
				name   string
				encode func(string) []uint32
			}{
				{"reference", func(text string) []uint32 { return referenceEncodeTekken(tok, text) }},
				{"per_piece", func(text string) []uint32 { return referenceEncodeGPT2PieceBuffers(tok, text) }},
				{"workspace", tok.encodeGPT2BPE},
			}
			if os.Getenv("TEKKEN_BENCH_REVERSE") == "1" {
				slices.Reverse(encoders)
			}
			for _, encoder := range encoders {
				b.Run("encode/"+encoder.name, func(b *testing.B) {
					b.SetBytes(int64(len(corpus.text)))
					b.ReportAllocs()
					for b.Loop() {
						tekkenIDsSink = encoder.encode(corpus.text)
					}
				})
			}
		})
	}
}
