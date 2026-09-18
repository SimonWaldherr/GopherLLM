package gopherllm

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/SimonWaldherr/GopherLLM/internal/jsonconstraint"
)

// ErrStructuredOutputIncomplete accompanies a truncated JSON prefix. A nil
// generation error in JSON mode guarantees a complete syntactically valid object.
var ErrStructuredOutputIncomplete = errors.New("structured output incomplete")
var ErrNoValidJSONToken = errors.New("no valid JSON continuation in model vocabulary")

// WithJSONObject constrains sampling to a JSON object grammar. Stops, active
// tools and speculative decoding are incompatible and rejected. Streams carry
// JSON prefixes; only successful completion guarantees a complete object.
func WithJSONObject() GenOption { return func(o *GenerationOptions) { o.JSONObject = true } }

type jsonTokenConstraint struct {
	state  jsonconstraint.State
	pieces []string
}

func newJSONTokenConstraint(tok *Tokenizer) *jsonTokenConstraint {
	c := &jsonTokenConstraint{pieces: make([]string, len(tok.Vocab))}
	for i := range c.pieces {
		c.pieces[i] = tok.DecodeToken(uint32(i))
	}
	return c
}
func (c *jsonTokenConstraint) mask(ctx context.Context, logits []float32, stop func(uint32) bool) error {
	any := false
	for i := range logits {
		if i%256 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		valid := false
		if i < len(c.pieces) && !stop(uint32(i)) && c.pieces[i] != "" {
			_, valid = c.state.Advance(c.pieces[i])
		}
		if !finite32(logits[i]) {
			valid = false
		}
		if !valid {
			logits[i] = float32(math.Inf(-1))
		} else if !math.IsNaN(float64(logits[i])) && !math.IsInf(float64(logits[i]), -1) {
			any = true
		}
	}
	if !any {
		return ErrNoValidJSONToken
	}
	return nil
}
func (c *jsonTokenConstraint) accept(token uint32) error {
	next, ok := c.state.Advance(c.pieces[token])
	if !ok {
		return fmt.Errorf("JSON constraint rejected sampled token")
	}
	c.state = next
	return nil
}
