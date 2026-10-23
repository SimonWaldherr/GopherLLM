package coreml

import (
	"fmt"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// GGUFTokenizer retains only tokenizer metadata, not model weights. Use a GGUF
// from the same checkpoint as the exported Core ML encoder.
type GGUFTokenizer struct {
	*gopherllm.Tokenizer
	PadToken uint32
}

// LoadTokenizer parses a GGUF through a temporary read-only mapping. It does
// not construct a Runner, dequantize tensors or retain the weight mapping.
func LoadTokenizer(path string) (*GGUFTokenizer, error) {
	mapped, err := gopherllm.OpenMmap(path)
	if err != nil {
		return nil, fmt.Errorf("coreml tokenizer: %w", err)
	}
	defer mapped.Close()
	gguf, err := gopherllm.ParseGGUFQuiet(mapped.Bytes())
	if err != nil {
		return nil, fmt.Errorf("coreml tokenizer: %w", err)
	}
	tok, err := gopherllm.TokenizerFromMetadata(gguf.Metadata)
	if err != nil {
		return nil, fmt.Errorf("coreml tokenizer: %w", err)
	}
	pad := gguf.GetU32("tokenizer.ggml.padding_token_id", 0)
	if uint64(pad) >= uint64(tok.VocabSize()) {
		return nil, fmt.Errorf("coreml tokenizer: padding token %d exceeds vocabulary size %d", pad, tok.VocabSize())
	}
	return &GGUFTokenizer{Tokenizer: tok, PadToken: pad}, nil
}
