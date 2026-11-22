package hestia

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/SimonWaldherr/GopherLLM/cmd/hestia/internal/hestia/adapters"
)

// ExecuteSingle applies one validated action to exactly one device and
// reads its state back to confirm the effect (CONCEPT.md section 7, rules
// 3, 6, 7). It never accepts a relative/toggle action: callers must have
// already resolved "etwas dunkler" into a concrete absolute value.
func ExecuteSingle(ctx context.Context, adapter adapters.Adapter, device Device, action string, args map[string]any) (adapters.State, error) {
	if device.ID == "" {
		return nil, fmt.Errorf("executor: device has no ID; refusing an implicit/global target")
	}
	if _, err := adapter.State(ctx, device.ID); err != nil {
		return nil, fmt.Errorf("executor: reading state before action: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := adapter.Apply(ctx, device.ID, action, args); err != nil {
		if errors.Is(err, adapters.ErrApplyOutcomeUnknown) {
			return nil, fmt.Errorf("executor: %w: %v", ErrOutcomeUnknown, err)
		}
		return nil, fmt.Errorf("executor: applying %s to %s: %w", action, device.ID, err)
	}
	after, err := adapter.State(ctx, device.ID)
	if err != nil {
		// The action may have been accepted by the device even though the
		// confirming read failed -- this is exactly the outcome_unknown
		// case from CONCEPT.md section 7, not a plain error the caller
		// should retry blindly.
		return nil, fmt.Errorf("executor: action applied but state readback failed: %w: %v", ErrOutcomeUnknown, err)
	}
	if !confirmedState(after, action, args) {
		return nil, fmt.Errorf("executor: readback has not confirmed the requested state: %w", ErrOutcomeUnknown)
	}
	return after, nil
}

func confirmedState(state adapters.State, action string, args map[string]any) bool {
	switch action {
	case "set_power":
		power, ok := state["power"].(bool)
		return ok && power == args["power"]
	case "set_brightness":
		want, ok := percentage(args["brightness"])
		if !ok {
			return false
		}
		if want == 0 {
			return state["power"] == false
		}
		got, ok := percentage(state["brightness"])
		return ok && state["power"] == true && math.Abs(got-want) <= 0.5
	default:
		return false
	}
}

func percentage(value any) (float64, bool) {
	switch n := value.(type) {
	case int:
		return float64(n), true
	case float64:
		return n, true
	default:
		return 0, false
	}
}

// ExecuteGroup applies one action to every device in targets, individually,
// and returns per-device errors rather than aborting on the first failure
// -- a partial group failure must stay visible per device, not collapse
// into one opaque error.
func ExecuteGroup(ctx context.Context, adapter adapters.Adapter, targets []Device, action string, args map[string]any) map[string]error {
	results := make(map[string]error, len(targets))
	for _, d := range targets {
		_, err := ExecuteSingle(ctx, adapter, d, action, args)
		results[d.ID] = err
	}
	return results
}
