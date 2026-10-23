//go:build !coreml

package gopherllm

import (
	"context"

	nativecoreml "github.com/SimonWaldherr/GopherLLM/internal/coreml"
)

// CoreMLFFNOptions selects devices for explicitly exported prefill FFNs.
// Build with CGO_ENABLED=1 and -tags coreml on an Apple platform to use them.
type CoreMLFFNOptions struct {
	CPUOnly bool
}

type coreMLFFNSet map[int]struct{}

func (coreMLFFNSet) close() {}

// SetCoreMLFFN requires a coreml build to load optional prefill FFN exports.
// An empty selection is accepted. Other builds do not import the source-hash
// implementation or any native framework into the inference package.
func (r *Runner) SetCoreMLFFN(manifests []string, opts CoreMLFFNOptions) error {
	if r == nil {
		return ErrRunnerClosed
	}
	if err := r.acquireModelLease(); err != nil {
		return err
	}
	defer r.releaseModelLease()
	if len(manifests) != 0 {
		return nativecoreml.ErrUnavailable
	}
	return nil
}

func tryCoreMLFFN(context.Context, *DecodeBuffer, int, []float32, []float32, int) (bool, error) {
	return false, nil
}
