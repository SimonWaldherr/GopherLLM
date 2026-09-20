package hestia

import "strings"

// Device is one controllable thing in the inventory: stable ID, display
// name, room, confirmed aliases, capabilities and the last observed state.
// Fresh state always comes from a device query (see Adapter), never from
// retrieval documents -- see CONCEPT.md section 11.
type Device struct {
	ID       string
	Name     string
	Room     string
	Aliases  []string // confirmed aliases only; see CONCEPT.md section 2 "Korrektur"
	Kind     string   // e.g. "light"
	State    map[string]any
	StateAt  string // RFC3339, when State was last observed
	StateSrc string // "device_readback" or similar; never "assumed"
}

// nameMatches reports whether q (already lowercased, trimmed) exactly
// matches this device's name or one of its confirmed aliases. Exact-match
// only: fuzzy similarity may surface candidates for a clarification, but
// must never by itself authorize a single-target write (CONCEPT.md
// section 7, "Der Resolver").
func (d Device) nameMatches(q string) bool {
	if strings.EqualFold(strings.TrimSpace(d.Name), q) {
		return true
	}
	for _, a := range d.Aliases {
		if strings.EqualFold(strings.TrimSpace(a), q) {
			return true
		}
	}
	return false
}

func (d Device) roomMatches(q string) bool {
	return q == "" || strings.EqualFold(strings.TrimSpace(d.Room), q)
}

// deviceKindWords maps a device Kind to the generic German nouns that name
// its category rather than any specific device ("Lampe"/"Licht"/"Leuchte"
// for "light"). CONCEPT.md section 2's own example -- "Mach die Lampe im
// Wohnzimmer an" must be ambiguous between Steh- and Deckenlampe -- needs
// this: a category word is not a specific device's name or alias, but it
// is still an exact, enumerable match (never fuzzy/partial), so it can only
// ever widen a candidate set into a clarification, never narrow it into a
// silently wrong single pick.
var deviceKindWords = map[string][]string{
	"light": {"lampe", "licht", "leuchte"},
}

func (d Device) kindMatchesWord(word string) bool {
	for _, w := range deviceKindWords[d.Kind] {
		if w == word {
			return true
		}
	}
	return false
}

// Inventory is the server-side, per-request-visible set of devices a
// resolver is allowed to consider. A model-mentioned device identifier is
// not itself a permission (CONCEPT.md section 7) -- callers must filter to
// what the requesting person may act on before passing an Inventory here.
type Inventory struct {
	Devices []Device
}

// FindByNameAndRoom returns every device whose name or alias exactly
// matches namePhrase (case-insensitive), optionally narrowed by room. If
// no device's own name/alias matches but namePhrase is a recognized device
// *category* word ("Lampe" for any light), every device of that category
// in the room is returned instead -- still an exact, enumerable match, so
// this can only widen a candidate set (surfacing an ambiguity for
// clarification) and never silently resolve to an unrelated device.
func (inv Inventory) FindByNameAndRoom(namePhrase, roomPhrase string) []Device {
	name := strings.ToLower(strings.TrimSpace(namePhrase))
	room := strings.ToLower(strings.TrimSpace(roomPhrase))
	var out []Device
	for _, d := range inv.Devices {
		if d.nameMatches(name) && d.roomMatches(room) {
			out = append(out, d)
		}
	}
	if len(out) > 0 {
		return out
	}
	for _, d := range inv.Devices {
		if d.kindMatchesWord(name) && d.roomMatches(room) {
			out = append(out, d)
		}
	}
	return out
}

// FindByRoom returns every device in a room, for an explicit group scope
// ("alle Lampen im Wohnzimmer"). Never used as a silent fallback for an
// ambiguous single-device request (CONCEPT.md section 3, section 7 rule 1).
func (inv Inventory) FindByRoom(roomPhrase string) []Device {
	room := strings.ToLower(strings.TrimSpace(roomPhrase))
	var out []Device
	for _, d := range inv.Devices {
		if d.roomMatches(room) && room != "" {
			out = append(out, d)
		}
	}
	return out
}
