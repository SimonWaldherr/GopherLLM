package hestia

import (
	"regexp"
	"strconv"
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
	// Explicit search commands are handled before device or document rules.
	// The query is user text and is never interpreted as a second command.
	if m := reWikipedia.FindStringSubmatch(t); m != nil {
		return Proposal{Intent: IntentWikipediaSearch, TargetPhrase: strings.TrimSpace(m[1])}
	}
	if m := reWebSearch.FindStringSubmatch(t); m != nil {
		return Proposal{Intent: IntentWebSearch, TargetPhrase: strings.TrimSpace(m[1])}
	}

	// Negation / self-correction: never let a partially-matched action
	// phrase followed by "nein"/"doch nicht" reach device_action.
	if strings.Contains(lower, "nein") || strings.Contains(lower, "doch nicht") {
		return Proposal{Intent: IntentCancel}
	}

	// A question about how to do something explains, it does not act.
	if strings.HasPrefix(lower, "wie ") || strings.Contains(lower, "wie würde") {
		return Proposal{Intent: IntentConversation}
	}

	if m := reAlleRaum.FindStringSubmatch(lower); m != nil {
		return Proposal{
			Intent: IntentDeviceAction, Action: "set_power", Scope: "group",
			RoomPhrase: m[1], Arguments: map[string]any{"power": m[2] != "aus"},
		}
	}
	if m := reDimme.FindStringSubmatch(t); m != nil {
		brightness, _ := strconv.Atoi(m[3]) // regexp accepts at most three decimal digits
		if brightness <= 100 {
			return Proposal{
				Intent: IntentDeviceAction, Action: "set_brightness", Scope: "single",
				TargetPhrase: strings.TrimSpace(m[1]), RoomPhrase: strings.ToLower(m[2]),
				Arguments: map[string]any{"brightness": brightness},
			}
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

var (
	reAlleRaum  = regexp.MustCompile(`^(?:(?:schalte|mach)\s+)?alle\s+(?:lampen|lichter)\s+im\s+([a-zäöüß]+)\s+(ein|an|aus)[.!]?$`)
	reDimme     = regexp.MustCompile(`(?i)^dimme\s+(?:die|den|das)\s+(.+?)(?:\s+im\s+([a-zäöüß]+))?\s+auf\s+([0-9]{1,3})\s*(?:prozent|%)?[.!]?$`)
	reWikipedia = regexp.MustCompile(`(?is)^(?:wikipedia\s*:\s*|suche\s+auf\s+wikipedia\s+nach(?:\s+|$))(.*)$`)
	reWebSearch = regexp.MustCompile(`(?is)^(?:websuche\s*:\s*|suche\s+im\s+(?:internet|web)\s+nach(?:\s+|$))(.*)$`)
	reSchalte   = regexp.MustCompile(`(?i)^schalte\s+(?:die|den|das)\s+(.+?)\s*(?:im\s+([a-zäöüß]+))?\s*(ein|aus)\.?$`)
	reMach      = regexp.MustCompile(`(?i)^mach\s+(?:die|den|das)\s+(.+?)\s*(?:im\s+([a-zäöüß]+))?\s*(an|aus)\.?$`)
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
