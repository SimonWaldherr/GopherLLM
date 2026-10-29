package tokenizer

import (
	"math/rand"
	"strings"
	"testing"
	"unicode"
)

func TestGPT2MappingWorkspaceMatchesPerPieceBuffers(t *testing.T) {
	var alphabet strings.Builder
	for i := 0; i < 256; i++ {
		alphabet.WriteByte(byte(i))
	}
	for _, pre := range []string{"", "qwen35", "tekken", "kimi"} {
		t.Run(pre, func(t *testing.T) {
			tok := randomGPT2Vocab(rand.New(rand.NewSource(53)), alphabet.String()+" wordBuilder Text", 0.5, 128)
			tok.Pre = pre
			for _, text := range []string{
				"", "Erkläre mir bitte HTTPServer2.", "a\xff\xfe\x00 日本語",
				strings.Repeat("Grüße aus Köln! L'été à Paris; España. ", 128),
				strings.Repeat(tekkenUniqueCorpus(), 8),
				strings.Repeat("日本語", 8192),
				"prefix " + strings.Repeat("日本語", 8192),
			} {
				got := tok.encodeGPT2BPE(text)
				want := referenceEncodeGPT2PieceBuffers(tok, text)
				if !equalIDs(got, want) {
					t.Fatalf("%d-byte input token IDs differ", len(text))
				}
			}
		})
	}
}

func TestGPT2MappingWorkspaceRespectsMutableByteEncoder(t *testing.T) {
	// Public custom byte maps may omit bytes, encode NUL or contain runes
	// that WriteRune replaces with U+FFFD. None can be treated as sentinels.
	tok := &Tokenizer{
		Mode: TokenizerGPT2BPE, Pre: "tekken",
		ByteEncoder: map[byte]rune{'a': 0, 'b': -1, 'c': unicode.MaxRune + 1, 'd': '𐐀'},
		TokenToID:   map[string]uint32{"\x00": 1, "�": 2, "𐐀": 3, "Z": 4},
	}
	text := strings.Repeat("abcde ", 20000)
	for _, mapping := range []map[byte]rune{
		tok.ByteEncoder,
		{'a': 'Z', 'b': '𐐀', 'e': 0},
		nil,
	} {
		tok.ByteEncoder = mapping
		if got, want := tok.encodeGPT2BPE(text), referenceEncodeGPT2PieceBuffers(tok, text); !equalIDs(got, want) {
			t.Fatal("custom byte map token IDs differ")
		}
	}
}
