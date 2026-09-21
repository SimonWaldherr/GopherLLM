package gopherllm

import (
	"context"
	"fmt"
	"math"
	"strings"
)

// ParakeetDecodeTokens renders TDT-decoded token IDs back to text. Unlike
// every text-generation tokenizer in this codebase, a transducer's output
// never needs an encode-side algorithm (there is no text prompt to
// tokenize -- only IDs a greedy search already chose), so this is a plain
// SentencePiece piece lookup and "▁" (word-boundary marker) ->
// " " substitution, not a Tokenizer-mode dispatch.
func ParakeetDecodeTokens(vocab []string, tokens []int) (string, error) {
	var sb strings.Builder
	for _, id := range tokens {
		if id < 0 || id >= len(vocab) {
			return "", fmt.Errorf("parakeet decode: token id %d out of vocabulary range [0,%d)", id, len(vocab))
		}
		sb.WriteString(strings.ReplaceAll(vocab[id], "▁", " "))
	}
	return strings.TrimPrefix(sb.String(), " "), nil
}

// TranscribeParakeet transcribes 16 kHz mono PCM using a local Parakeet-TDT
// GGUF (general.architecture=="asr", asr.head_type=="tdt"). See parakeet.go's
// doc comment for the architecture this loads and runs, and this package's
// other parakeet_*.go files for each stage's implementation.
//
// Caveat this function's caller should know: unlike the Voxtral loaders in
// this codebase (each cross-validated component-by-component against a
// second, independently-converted file of the identical checkpoint, down to
// byte-identical intermediate tensors), this implementation has no such
// second reference to diff against -- Parakeet-TDT is only known to this
// codebase through one community GGUF conversion. Every formula was
// implemented against NVIDIA NeMo's own published source
// (features.py/rnnt.py) rather than guessed, and it has been verified
// end-to-end: two known English utterances transcribed correctly
// (TestTranscribeParakeetRealAudio), including exact punctuation and
// capitalization on the longer one. The one remaining flagged unknown is
// the joint network's activation (relu, chosen as NeMo's common published
// default; the checkpoint's own training config isn't recoverable from
// this GGUF) -- it produced correct output on both test clips regardless,
// but hasn't been independently confirmed against this specific
// checkpoint's actual training configuration or a from-scratch NeMo run.
func TranscribeParakeet(modelPath string, samples []float32, logw func(string, ...any)) (string, error) {
	if len(samples) == 0 {
		return "", fmt.Errorf("audio is empty")
	}
	for _, s := range samples {
		if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
			return "", fmt.Errorf("audio contains non-finite samples")
		}
	}
	model, err := OpenParakeet(modelPath, nil)
	if err != nil {
		return "", err
	}
	defer model.Close()
	return model.TranscribeOffline(context.Background(), samples, logw)
}
