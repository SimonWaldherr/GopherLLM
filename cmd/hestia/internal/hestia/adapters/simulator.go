package adapters

import (
	"context"
	"fmt"
	"sync"
)

// Simulator is an in-memory Adapter for development and demos (CONCEPT.md
// section 3: "Ein Simulator unterstützt Entwicklung und Vorführung").
// Device control only counts as done with a real, configured, verified
// adapter -- the simulator is explicitly not that.
type Simulator struct {
	mu    sync.Mutex
	byID  map[string]Entity
	state map[string]State
}

// NewSimulator creates a Simulator seeded with the given entities, each
// starting powered off.
func NewSimulator(entities ...Entity) *Simulator {
	s := &Simulator{
		byID:  make(map[string]Entity, len(entities)),
		state: make(map[string]State, len(entities)),
	}
	for _, e := range entities {
		s.byID[e.ID] = e
		s.state[e.ID] = State{"power": false}
	}
	return s
}

func (s *Simulator) Inventory(ctx context.Context) ([]Entity, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Entity, 0, len(s.byID))
	for _, e := range s.byID {
		out = append(out, e)
	}
	return out, nil
}

func (s *Simulator) State(ctx context.Context, entityID string) (State, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	st, ok := s.state[entityID]
	if !ok {
		return nil, fmt.Errorf("simulator: unknown entity %q", entityID)
	}
	out := make(State, len(st))
	for k, v := range st {
		out[k] = v
	}
	return out, nil
}

// Apply supports "set_power" (bool) and "set_brightness" (0-100 int/float)
// -- the two capabilities CONCEPT.md section 3 scopes for the first
// version. Any other action is refused rather than silently accepted.
func (s *Simulator) Apply(ctx context.Context, entityID, action string, args map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.byID[entityID]; !ok {
		return fmt.Errorf("simulator: unknown entity %q", entityID)
	}
	st := s.state[entityID]
	if st == nil {
		st = State{}
	}
	switch action {
	case "set_power":
		v, ok := args["power"].(bool)
		if !ok {
			return fmt.Errorf("simulator: set_power requires a bool \"power\" argument")
		}
		st["power"] = v
	case "set_brightness":
		v, ok := numericArg(args["brightness"])
		if !ok || v < 0 || v > 100 {
			return fmt.Errorf("simulator: set_brightness requires a numeric \"brightness\" argument in [0,100]")
		}
		st["brightness"] = v
		st["power"] = v > 0
	default:
		return fmt.Errorf("simulator: unsupported action %q", action)
	}
	s.state[entityID] = st
	return nil
}

func numericArg(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	default:
		return 0, false
	}
}
