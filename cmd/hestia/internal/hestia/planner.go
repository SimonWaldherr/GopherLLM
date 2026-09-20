package hestia

import (
	"regexp"
	"strings"
)

// ParseTextProposal turns finalized German text into a Proposal using
// simple keyword/regex rules.
//
// This is an explicit placeholder for CONCEPT.md section 7's real planner
// (a local language model returning the same closed schema via GopherLLM).
// It exists so the resolver/policy/executor/journal chain -- the
// safety-relevant part this milestone is actually about -- can be built and
// tested against real proposals today. Replacing this function's body with
// a GopherLLM.Model.Chat call using the same Proposal shape is the planned
// next step; nothing downstream of Proposal needs to change for that swap.
func ParseTextProposal(text string) Proposal {
	t := strings.TrimSpace(text)
	lower := strings.ToLower(t)

	// Negation / self-correction: never let a partially-matched action
	// phrase followed by "nein"/"doch nicht" reach device_action.
	if strings.Contains(lower, "nein") || strings.Contains(lower, "doch nicht") {
		return Proposal{Intent: IntentCancel}
	}

	// A question about how to do something explains, it does not act.
	if strings.HasPrefix(lower, "wie ") || strings.Contains(lower, "wie würde") {
		return Proposal{Intent: IntentConversation}
	}

	if on, off, ok := matchGroupCommand(lower); ok {
		room := extractRoom(lower)
		power := on
		_ = off
		return Proposal{
			Intent: IntentDeviceAction, Action: "set_power", Scope: "group",
			RoomPhrase: room, Arguments: map[string]any{"power": power},
		}
	}

	if target, room, power, ok := matchSingleCommand(t); ok {
		return Proposal{
			Intent: IntentDeviceAction, Action: "set_power", Scope: "single",
			TargetPhrase: target, RoomPhrase: room, Arguments: map[string]any{"power": power},
		}
	}

	if strings.Contains(lower, "was sagen") || strings.Contains(lower, "unterlagen") || strings.Contains(lower, "dokument") {
		return Proposal{Intent: IntentKnowledgeQuery, TargetPhrase: t}
	}

	return Proposal{Intent: IntentConversation}
}

var reAlleRaum = regexp.MustCompile(`alle\s+(\w+)\s+im\s+([a-zäöüß]+)`)

func matchGroupCommand(lower string) (on, off, ok bool) {
	m := reAlleRaum.FindStringSubmatch(lower)
	if m == nil {
		return false, false, false
	}
	if strings.Contains(lower, " aus") {
		return false, true, true
	}
	return true, false, true
}

func extractRoom(lower string) string {
	if m := reAlleRaum.FindStringSubmatch(lower); m != nil {
		return m[2]
	}
	if m := reImRaum.FindStringSubmatch(lower); m != nil {
		return m[1]
	}
	return ""
}

var (
	reSchalte = regexp.MustCompile(`(?i)^schalte\s+(?:die|den|das)\s+(.+?)\s*(?:im\s+([a-zäöüß]+))?\s*(ein|aus)\.?$`)
	reMach    = regexp.MustCompile(`(?i)^mach\s+(?:die|den|das)\s+(.+?)\s*(?:im\s+([a-zäöüß]+))?\s*(an|aus)\.?$`)
	reImRaum  = regexp.MustCompile(`im\s+([a-zäöüß]+)`)
)

// matchSingleCommand extracts (target device phrase, room phrase, desired
// power) from one of the two accepted verb forms. Both regexes carry the
// (?i) flag and run against the original-cased text, so a device name's
// own casing survives into target (matching against Inventory is
// case-insensitive regardless -- see Device.nameMatches).
func matchSingleCommand(original string) (target, room string, power bool, ok bool) {
	if m := reSchalte.FindStringSubmatch(original); m != nil {
		return strings.TrimSpace(m[1]), strings.ToLower(m[2]), strings.EqualFold(m[3], "ein"), true
	}
	if m := reMach.FindStringSubmatch(original); m != nil {
		return strings.TrimSpace(m[1]), strings.ToLower(m[2]), strings.EqualFold(m[3], "an"), true
	}
	return "", "", false, false
}
