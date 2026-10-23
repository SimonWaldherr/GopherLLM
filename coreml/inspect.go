package coreml

import backend "github.com/SimonWaldherr/GopherLLM/internal/coreml"

// Plan counts graph operations by their preferred device in Core ML's compute
// plan. It describes planned placement, not observed utilization or time share.
type Plan = backend.Plan

// Inspect loads a diagnostic compute plan using the same device permission as
// OpenEmbedder. It does not predict or keep another model instance alive.
// Requires macOS 14.4/iOS 17.4 or newer in addition to the coreml build tag.
func Inspect(path string, units ComputeUnits) (Plan, error) {
	return backend.Inspect(path, backend.Config{ComputeUnits: units})
}
