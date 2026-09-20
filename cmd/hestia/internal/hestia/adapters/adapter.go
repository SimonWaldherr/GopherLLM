// Package adapters holds the narrow device-adapter interface (CONCEPT.md
// section 11: "Die schmale interne Adapter-Schnittstelle") and its
// implementations. An adapter must take a concrete entity ID -- an empty or
// implicit global target is refused by the caller before it ever reaches
// an adapter.
package adapters

import "context"

// State is a device's reported property set, e.g. {"power": true}.
type State map[string]any

// Adapter is the minimal device integration surface: inventory, state
// query, and applying one already-validated action. Credentials and
// network addresses are configured administratively and never appear in
// arguments passed here.
type Adapter interface {
	// Inventory lists the entities this adapter currently knows about.
	Inventory(ctx context.Context) ([]Entity, error)
	// State queries one entity's current reported state.
	State(ctx context.Context, entityID string) (State, error)
	// Apply performs one validated, concrete action against one entity.
	// The caller re-queries State afterward to confirm the effect -- a
	// successful Apply return is not itself proof of the new device state
	// (CONCEPT.md section 7, rule 7).
	Apply(ctx context.Context, entityID, action string, args map[string]any) error
}

// Entity is one device as reported by an adapter's own inventory.
type Entity struct {
	ID   string
	Name string
	Room string
	Kind string
}
