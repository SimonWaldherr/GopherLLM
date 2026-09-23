package tokenizer

// BuildByteMaps returns the GPT-2 byte/rune mapping used by byte-level BPE.
func BuildByteMaps() (map[byte]rune, map[rune]byte) { return buildByteMaps() }

// PreTokenizerGPT2 exposes the GPT-2 text splitting rule to package adapters.
func PreTokenizerGPT2(text string) []string { return pretokenizeGPT2(text) }

// PreTokenizerKimi exposes Kimi's GPT-2-family splitting rule.
func PreTokenizerKimi(text string) []string { return pretokenizeKimi(text) }

// PreTokenizerQwen35 exposes Qwen 3.5's GPT-2-family splitting rule.
func PreTokenizerQwen35(text string) []string { return pretokenizeQwen35(text) }

// PreTokenizerTekken exposes Mistral Tekken's GPT-2-family splitting rule.
func PreTokenizerTekken(text string) []string { return pretokenizeTekken(text) }

// VoxtralTokenizerFromTekkenJSON builds a tokenizer from the official Tekken JSON format.
func VoxtralTokenizerFromTekkenJSON(data []byte) (*Tokenizer, error) {
	return voxtralTokenizerFromTekkenJSON(data)
}
