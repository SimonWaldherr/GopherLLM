package gopherllm

import "github.com/SimonWaldherr/GopherLLM/internal/tokenizer"

type TokenizerMode = tokenizer.TokenizerMode
type Pair = tokenizer.Pair
type Tokenizer = tokenizer.Tokenizer

const (
	TokenizerSentencePiece = tokenizer.TokenizerSentencePiece
	TokenizerGPT2BPE       = tokenizer.TokenizerGPT2BPE
	TokenizerWordPiece     = tokenizer.TokenizerWordPiece
)

// TokenizerFromMetadata builds a tokenizer from GGUF tokenizer metadata.
func TokenizerFromMetadata(metadata map[string]MetaValue) (*Tokenizer, error) {
	return tokenizer.TokenizerFromMetadata(metadata)
}

// These helpers retain package-local names used by existing benchmarks and
// tokenizer fixtures while their implementations live in internal/tokenizer.
func buildByteMaps() (map[byte]rune, map[rune]byte) { return tokenizer.BuildByteMaps() }
func pretokenizeGPT2(text string) []string          { return tokenizer.PreTokenizerGPT2(text) }
func pretokenizeKimi(text string) []string          { return tokenizer.PreTokenizerKimi(text) }
func pretokenizeQwen35(text string) []string        { return tokenizer.PreTokenizerQwen35(text) }
func pretokenizeTekken(text string) []string        { return tokenizer.PreTokenizerTekken(text) }
func voxtralTokenizerFromTekkenJSON(data []byte) (*Tokenizer, error) {
	return tokenizer.VoxtralTokenizerFromTekkenJSON(data)
}
