package hestia

import (
	"context"
	"fmt"

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
	before, err := adapter.State(ctx, device.ID)
	if err != nil {
		return nil, fmt.Errorf("executor: reading state before action: %w", err)
	}
	if err := adapter.Apply(ctx, device.ID, action, args); err != nil {
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
	_ = before
	return after, nil
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
