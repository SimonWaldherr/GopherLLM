package gopherllm

import (
	"context"
	"fmt"
	"io"
	"math"
	"strings"
	"time"
)

const (
	voxtralLeftPadTokens  = 32
	voxtralRightPadTokens = 17
)

// TranscribeVoxtralRealtime transcribes 16 kHz mono PCM using a local GGUF.
// Model mappings and weights are owned by this call and released on return.
// Cancellation is checked between encoder frames/layers and decoder steps.
// maxExtraSteps adds optional right-padding tokens (normally zero). Decoding
// always stays within the encoded audio span, matching the reference schedule.
func TranscribeVoxtralRealtime(ctx context.Context, modelPath string, samples []float32, maxExtraSteps int, logw io.Writer) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if len(samples) == 0 {
		return "", fmt.Errorf("audio is empty")
	}
	if maxExtraSteps < 0 || maxExtraSteps > 256 {
		return "", fmt.Errorf("max extra steps must be between 0 and 256")
	}
	for _, sample := range samples {
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			return "", fmt.Errorf("audio contains non-finite samples")
		}
	}
	if logw == nil {
		logw = io.Discard
	}
	mmap, err := OpenMmap(modelPath)
	if err != nil {
		return "", fmt.Errorf("opening model: %w", err)
	}
	defer mmap.Close()
	data := mmap.Bytes()

	gguf, err := ParseGGUF(data)
	if err != nil {
		return "", fmt.Errorf("parsing GGUF: %w", err)
	}

	cfg, w, err := LoadVoxtralRealtimeModel(data, gguf, false, logw)
	defer releaseVoxtralRealtimeWeights(&w)
	if err != nil {
		return "", fmt.Errorf("loading Voxtral Realtime model: %w", err)
	}

	tok, err := TokenizerFromMetadata(gguf.Metadata)
	if err != nil {
		return "", fmt.Errorf("building tokenizer: %w", err)
	}

	if cfg.Mel.SampleRate != 16000 {
		return "", fmt.Errorf("Voxtral requires a 16 kHz frontend, got %d", cfg.Mel.SampleRate)
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}

	// rawAudioLengthPerTok: samples of raw audio one adapter/decoder-aligned
	// embedding covers -- Mel.HopLength samples per mel frame times how many
	// mel frames feed one adapter output (the conv stem's stride-2 layer
	// times the adapter's own downsample factor). Verified equal to the
	// reference's hardcoded RAW_AUDIO_LENGTH_PER_TOK=1280 for this checkpoint
	// (160 * 2 * 4), but derived from metadata here instead of hardcoded so
	// a differently-configured Voxtral Realtime checkpoint still lines up.
	rawAudioLengthPerTok := cfg.Mel.HopLength * 2 * cfg.Projector.DownsampleFactor
	padded := padVoxtralAudioOffline(samples, rawAudioLengthPerTok)
	padded = append(padded, make([]float32, maxExtraSteps*rawAudioLengthPerTok)...)
	fmt.Fprintf(logw, "padded to %d samples (left=%d tokens, right=%d+align tokens worth of silence)\n", len(padded), voxtralLeftPadTokens, voxtralRightPadTokens)

	start := time.Now()
	audioEmbeds, err := EncodeAudioVoxtralRealtimeContext(ctx, cfg, w, padded)
	if err != nil {
		return "", fmt.Errorf("encoding audio: %w", err)
	}
	fmt.Fprintf(logw, "encoded %d audio embedding groups in %s\n", len(audioEmbeds), time.Since(start))

	state, err := NewVoxtralRealtimeDecoderState(cfg, w, cfg.Time.DefaultNumDelayTokens)
	if err != nil {
		return "", fmt.Errorf("creating decoder state: %w", err)
	}

	eosID := int(gguf.GetU32("tokenizer.ggml.eos_token_id", 2))
	bosID := int(gguf.GetU32("tokenizer.ggml.bos_token_id", 1))
	padID := cfg.StreamingPadTokenID

	// Prefill: BOS followed by (voxtralLeftPadTokens + DefaultNumDelayTokens) pad
	// tokens, each teacher-forced against the correspondingly early
	// (silence-derived, thanks to the left padding above) audio embedding --
	// see this file's doc comment. Only the LAST prefill position's logits
	// matter; every one before it just needs to run for its K/V side effect.
	promptIDs := make([]int, 1+voxtralLeftPadTokens+cfg.Time.DefaultNumDelayTokens)
	promptIDs[0] = bosID
	for i := 1; i < len(promptIDs); i++ {
		promptIDs[i] = padID
	}
	L := len(promptIDs)
	if L > len(audioEmbeds) {
		return "", fmt.Errorf("audio too short: need at least %d adapter frames for the prefill alone, got %d (try a longer clip)", L, len(audioEmbeds))
	}

	decodeStart := time.Now()
	var prevToken int
	for pos := range L {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		te, err := EmbedVoxtralRealtimeToken(cfg, w, promptIDs[pos])
		if err != nil {
			return "", fmt.Errorf("prefill pos %d: embedding token %d: %w", pos, promptIDs[pos], err)
		}
		fused := make([]float32, len(te))
		for i := range fused {
			fused[i] = te[i] + audioEmbeds[pos][i]
		}
		logits, err := ForwardVoxtralRealtimeDecoderStep(cfg, w, state, fused, pos)
		if err != nil {
			return "", fmt.Errorf("prefill pos %d: decoder step: %w", pos, err)
		}
		if pos == L-1 {
			prevToken = voxtralArgmax(logits)
		}
	}
	if logw != io.Discard {
		fmt.Fprintf(logw, "prefill done (%d positions), first generated token=%d %q\n", L, prevToken, tok.DecodeToken(uint32(prevToken)))
	}

	tokenTypes, _ := gguf.Metadata["tokenizer.ggml.token_type"].AsU32Array()
	var out strings.Builder
	if prevToken != padID && prevToken != eosID {
		out.WriteString(voxtralTranscriptPiece(tok, tokenTypes, prevToken))
	}

	totalSteps := len(audioEmbeds)
	for pos := L; pos < totalSteps && prevToken != eosID; pos++ {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		ae := audioEmbeds[pos]
		te, err := EmbedVoxtralRealtimeToken(cfg, w, prevToken)
		if err != nil {
			return "", fmt.Errorf("pos %d: embedding token %d: %w", pos, prevToken, err)
		}
		fused := make([]float32, len(te))
		for i := range fused {
			fused[i] = te[i] + ae[i]
		}

		logits, err := ForwardVoxtralRealtimeDecoderStep(cfg, w, state, fused, pos)
		if err != nil {
			return "", fmt.Errorf("pos %d: decoder step: %w", pos, err)
		}
		next := voxtralArgmax(logits)
		if logw != io.Discard {
			fmt.Fprintf(logw, "pos=%d token=%d %q\n", pos, next, tok.DecodeToken(uint32(next)))
		}
		if next == eosID {
			break
		}
		if next != padID {
			out.WriteString(voxtralTranscriptPiece(tok, tokenTypes, next))
		}
		prevToken = next
	}
	fmt.Fprintf(logw, "decoded in %s\n", time.Since(decodeStart))

	return strings.TrimSpace(out.String()), nil
}

// padVoxtralAudioOffline mirrors mistral_common's AudioEncoder.pad for
// offline (whole-clip, non-live) transcription: zero-pad
// voxtralLeftPadTokens*rawAudioLengthPerTok samples on the left, and
// (alignment remainder + voxtralRightPadTokens*rawAudioLengthPerTok) samples on
// the right so every adapter output frame is a complete, fully-covered
// group (see voxtral_realtime_audio.go's projectAudioVoxtralRealtime, which
// silently drops a ragged trailing group otherwise).
func padVoxtralAudioOffline(samples []float32, rawAudioLengthPerTok int) []float32 {
	leftPad := voxtralLeftPadTokens * rawAudioLengthPerTok
	alignPad := (rawAudioLengthPerTok - len(samples)%rawAudioLengthPerTok) % rawAudioLengthPerTok
	rightPad := alignPad + voxtralRightPadTokens*rawAudioLengthPerTok

	out := make([]float32, leftPad+len(samples)+rightPad)
	copy(out[leftPad:], samples)
	return out
}

func voxtralArgmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// Control markers (notably STREAMING_WORD) drive alignment but are not text.
// Decode byte pieces before concatenation so UTF-8 characters split across
// multiple tokens remain intact in the final string.
func voxtralTranscriptPiece(tok *Tokenizer, tokenTypes []uint32, id int) string {
	if id < 0 || id >= len(tok.Vocab) {
		return ""
	}
	if id < len(tokenTypes) && (tokenTypes[id] == 3 || tokenTypes[id] == 5) {
		return ""
	}
	switch tok.Vocab[id] {
	case "[STREAMING_PAD]", "[STREAMING_WORD]", "<s>", "</s>":
		return ""
	}
	return tok.DecodeToken(uint32(id))
}
