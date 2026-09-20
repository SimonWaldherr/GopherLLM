package hestia

import "fmt"

// ResolveOutcome is what the resolver decided for one proposal.
type ResolveOutcome int

const (
	// ResolveNone: no matching device at all -- report, do not guess.
	ResolveNone ResolveOutcome = iota
	// ResolveSingle: exactly one device matched -- may proceed.
	ResolveSingle
	// ResolveAmbiguous: more than one device matched a single-scope
	// request -- must ask a clarifying question, never guess or act on
	// all of them (CONCEPT.md section 7, rule 1).
	ResolveAmbiguous
	// ResolveGroup: an explicit group/room scope resolved to a concrete
	// list -- CONCEPT.md section 7, rule 2: requires confirmation in v1.
	ResolveGroup
)

// ResolveResult is the resolver's answer for one proposal.
type ResolveResult struct {
	Outcome    ResolveOutcome
	Candidates []Device // single: len==1; ambiguous: len>1; group: the resolved list
}

// Resolve maps a device_action proposal onto the caller-visible Inventory.
// It never falls back from a single target to "all devices of that kind":
// an unresolved or ambiguous single-scope request always comes back as
// ResolveNone/ResolveAmbiguous, never as an implicit group.
func Resolve(inv Inventory, p Proposal) (ResolveResult, error) {
	if p.Intent != IntentDeviceAction {
		return ResolveResult{}, fmt.Errorf("resolve: proposal intent %q is not device_action", p.Intent)
	}

	if p.Scope == "group" {
		if p.RoomPhrase == "" {
			return ResolveResult{}, fmt.Errorf("resolve: group scope requires an explicit room phrase")
		}
		matches := inv.FindByRoom(p.RoomPhrase)
		if len(matches) == 0 {
			return ResolveResult{Outcome: ResolveNone}, nil
		}
		return ResolveResult{Outcome: ResolveGroup, Candidates: matches}, nil
	}

	matches := inv.FindByNameAndRoom(p.TargetPhrase, p.RoomPhrase)
	switch len(matches) {
	case 0:
		return ResolveResult{Outcome: ResolveNone}, nil
	case 1:
		return ResolveResult{Outcome: ResolveSingle, Candidates: matches}, nil
	default:
		return ResolveResult{Outcome: ResolveAmbiguous, Candidates: matches}, nil
	}
}
