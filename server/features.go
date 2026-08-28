package server

import (
	"strconv"
	"strings"
)

// Features selects the optional capabilities a server exposes. The zero value
// is the profile a first-time user should get: chat, completions, and
// embeddings against the loaded model, and nothing that reaches the internet,
// rewrites process-wide state, or benchmarks the host.
//
// A capability that is off is not registered at all rather than merely hidden
// in the Web UI, so a caller who guesses the path gets a 404 instead of a
// permission check. The Web UI reads the enabled set from GET /deployment and
// only renders the panels the server actually backs.
//
// Hosts that want the pre-Features behavior back can pass AllFeatures().
type Features struct {
	// ModelCatalog serves GET /models and the /models/load hot-swap, plus
	// /models/architecture and the embedding-model routes. Loads stay
	// restricted to ModelDir; this only decides whether the routes exist.
	ModelCatalog bool
	// ModelDownload adds /models/search and /models/download. Both make
	// outbound Hugging Face requests and write GGUF files into the local
	// cache, so they are the one feature that turns a local chat server into a
	// downloader.
	ModelDownload bool
	// AutoTune adds /autotune and /autotune/run. A run saturates the machine
	// for the length of the calibration and then changes process-wide runtime
	// tuning for every later request.
	AutoTune bool
	// RemoteProxy adds /remote and /remote/models, which point every
	// subsequent completion at another endpoint. That is useful for comparing
	// a local model against a hosted one and damaging if anybody else can
	// reach it, since it redirects the whole conversation.
	RemoteProxy bool
	// WebLookup offers the Wikimedia and OpenStreetMap agentic tools to chat
	// requests. Individual requests still have to ask for them by name, so
	// this is the outer switch rather than the only one.
	WebLookup bool
	// Spreadsheet adds /batch/parse, which turns an uploaded CSV/TSV
	// spreadsheet into prompts for batch runs.
	Spreadsheet bool
}

// AllFeatures returns every optional capability enabled. It is the explicit
// opt-in for hosts that want the full surface, and what the CLI's --full flag
// selects.
func AllFeatures() Features {
	return Features{
		ModelCatalog:  true,
		ModelDownload: true,
		AutoTune:      true,
		RemoteProxy:   true,
		WebLookup:     true,
		Spreadsheet:   true,
	}
}

// featureNames maps the wire names accepted by ParseFeatures and reported by
// GET /deployment onto the struct. The wire names are stable API.
func (f *Features) fieldFor(name string) *bool {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "model-catalog", "models":
		return &f.ModelCatalog
	case "model-download", "download":
		return &f.ModelDownload
	case "autotune":
		return &f.AutoTune
	case "remote":
		return &f.RemoteProxy
	case "web-lookup", "web":
		return &f.WebLookup
	case "spreadsheet", "batch":
		return &f.Spreadsheet
	default:
		return nil
	}
}

// FeatureNames lists the canonical wire names in the order the CLI help and
// the Web UI present them.
func FeatureNames() []string {
	return []string{"model-catalog", "model-download", "autotune", "remote", "web-lookup", "spreadsheet"}
}

// ParseFeatures reads a comma-separated capability list, as accepted by the
// CLI's --enable flag. "all" selects everything. Unknown names are an error
// rather than a silent no-op: a typo must not look like a disabled feature.
func ParseFeatures(list string) (Features, error) {
	var f Features
	for _, raw := range strings.Split(list, ",") {
		name := strings.ToLower(strings.TrimSpace(raw))
		if name == "" {
			continue
		}
		if name == "all" {
			f = AllFeatures()
			continue
		}
		field := f.fieldFor(name)
		if field == nil {
			return Features{}, &UnknownFeatureError{Name: strings.TrimSpace(raw)}
		}
		*field = true
	}
	return f, nil
}

// UnknownFeatureError reports a capability name that ParseFeatures does not
// know, and lists the ones it does.
type UnknownFeatureError struct{ Name string }

func (e *UnknownFeatureError) Error() string {
	return "unknown feature " + strconv.Quote(e.Name) + " (known: " + strings.Join(FeatureNames(), ", ") + ", all)"
}

// status is the map GET /deployment embeds so the Web UI can render only the
// panels this server backs.
func (f Features) status() map[string]any {
	return map[string]any{
		"model-catalog":  f.ModelCatalog,
		"model-download": f.ModelDownload,
		"autotune":       f.AutoTune,
		"remote":         f.RemoteProxy,
		"web-lookup":     f.WebLookup,
		"spreadsheet":    f.Spreadsheet,
	}
}

// EnabledNames lists the enabled capabilities, in FeatureNames order. It is
// what the startup banner prints and what --print-config records.
func (f Features) EnabledNames() []string {
	var out []string
	for _, name := range FeatureNames() {
		if field := f.fieldFor(name); field != nil && *field {
			out = append(out, name)
		}
	}
	return out
}

// Merge returns the union of two feature sets, so repeated --enable flags add
// up instead of replacing each other.
func (f Features) Merge(o Features) Features {
	for _, name := range FeatureNames() {
		if field := o.fieldFor(name); field != nil && *field {
			*f.fieldFor(name) = true
		}
	}
	return f
}
