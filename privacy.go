package gopherllm

import (
	"encoding/json"
	"io"
)

// PrivacyReport makes GopherLLM's local-first behavior inspectable by people
// and deployment tooling. It deliberately describes network-capable features
// separately from normal inference: none are enabled implicitly.
type PrivacyReport struct {
	Inference      string              `json:"inference"`
	Telemetry      bool                `json:"telemetry"`
	DefaultNetwork string              `json:"default_network"`
	LocalStorage   []string            `json:"local_storage"`
	OptInFeatures  []PrivacyNetworkUse `json:"opt_in_features"`
}

// PrivacyNetworkUse describes data that may leave the machine only after a
// caller or user explicitly enables the named feature.
type PrivacyNetworkUse struct {
	Feature      string   `json:"feature"`
	Destinations []string `json:"destinations"`
	Data         string   `json:"data"`
}

// DefaultPrivacyReport returns the privacy contract shared by the CLI and
// HTTP server. Normal model loading and inference are local and telemetry-free.
func DefaultPrivacyReport() PrivacyReport {
	return PrivacyReport{
		Inference:      "local GGUF inference; prompts and generated text stay in-process",
		Telemetry:      false,
		DefaultNetwork: "none",
		LocalStorage: []string{
			"selected local model path and optional autotuning cache",
			"Hugging Face model cache only when an hf: selector is explicitly used",
		},
		OptInFeatures: []PrivacyNetworkUse{
			{Feature: "Hugging Face import", Destinations: []string{"huggingface.co or HF_ENDPOINT"}, Data: "requested repository, revision, GGUF filenames, and HF_TOKEN when configured"},
			{Feature: "Wikipedia and Wikidata research tools", Destinations: []string{"Wikipedia", "Wikidata", "Wikidata Query Service"}, Data: "only tool arguments selected by the model; never the full chat transcript"},
			{Feature: "OpenStreetMap place search", Destinations: []string{"Nominatim endpoint selected by the operator"}, Data: "only the place-search query selected by the model; do not submit personal or confidential data"},
			{Feature: "remote model proxy", Destinations: []string{"operator-configured remote endpoint"}, Data: "request content sent to that endpoint"},
		},
	}
}

// WritePrivacyReport writes a stable, indented JSON report suitable for
// `gopherllm --privacy`, logs, and deployment audits.
func WritePrivacyReport(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(DefaultPrivacyReport())
}
