package gopherllm

import (
	"context"
	"strings"
	"testing"
)

func TestPromptLookupOptionsValidation(t *testing.T) {
	base := DefaultGenerationOptions()
	if base.PromptLookupTokens != 0 {
		t.Fatal("prompt lookup is enabled by default")
	}
	for _, tc := range []struct {
		name        string
		lookup, mtp int
		temperature float32
		topK        int
		json        bool
		wantError   string
	}{
		{name: "disabled", temperature: 0.7, topK: 40},
		{name: "greedy temperature", lookup: 7, temperature: 0, topK: 40},
		{name: "greedy top-k", lookup: 4, temperature: 0.7, topK: 1},
		{name: "lookup and MTP", lookup: 7, mtp: 4, temperature: 0, topK: 40},
		{name: "negative", lookup: -1, topK: 1, wantError: "range [0, 7]"},
		{name: "too many", lookup: 8, topK: 1, wantError: "range [0, 7]"},
		{name: "stochastic", lookup: 4, temperature: 0.7, topK: 40, wantError: "requires greedy sampling"},
		{name: "constrained JSON", lookup: 4, topK: 1, json: true, wantError: "JSON output cannot"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := base
			opts.PromptLookupTokens, opts.MTPDraftTokens = tc.lookup, tc.mtp
			opts.Sampler.Temperature, opts.Sampler.TopK = tc.temperature, tc.topK
			opts.JSONObject = tc.json
			err := opts.Validate()
			if tc.wantError == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.wantError) {
				t.Fatalf("error = %v, want %q", err, tc.wantError)
			}
		})
	}
}

func TestWithPromptLookupTokens(t *testing.T) {
	opts := buildGenOptions(context.Background(), []GenOption{
		WithTemperature(0), WithMTPDraftTokens(4), WithPromptLookupTokens(7),
	})
	if opts.PromptLookupTokens != 7 || opts.MTPDraftTokens != 4 {
		t.Fatalf("options = %+v", opts)
	}
	if err := opts.Validate(); err != nil {
		t.Fatal(err)
	}
	WithPromptLookupTokens(0)(&opts)
	if opts.PromptLookupTokens != 0 || opts.MTPDraftTokens != 4 {
		t.Fatal("disabling lookup changed MTP")
	}
}
