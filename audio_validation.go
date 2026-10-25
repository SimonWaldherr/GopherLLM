package gopherllm

import (
	"context"
	"errors"

	"github.com/SimonWaldherr/GopherLLM/internal/numeric"
)

// validateAudioSamples accepts empty chunks, as streaming finalization can
// carry no new audio. Recording entry points enforce their own length limits.
func validateAudioSamples(samples []float32) error {
	if !numeric.AllFiniteF32(samples) {
		return errors.New("audio contains non-finite samples")
	}
	return nil
}

func validateVoxtralOfflineInput(ctx context.Context, samples []float32, maxExtraSteps int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(samples) == 0 {
		return errors.New("audio is empty")
	}
	if maxExtraSteps < 0 || maxExtraSteps > 256 {
		return errors.New("max extra steps must be between 0 and 256")
	}
	return validateAudioSamples(samples)
}
