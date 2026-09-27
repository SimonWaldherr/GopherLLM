package mobile

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

type loadOptions struct {
	Threads          int    `json:"threads"`
	PrepareQuantized bool   `json:"prepare_quantized"`
	OutOfCore        bool   `json:"out_of_core"`
	Prefault         string `json:"prefault"`
	Metal            bool   `json:"metal"`
}

// defaultLoadOptions pages weights in on demand ("none") rather than reading
// the whole file up front: mobile apps care more about launch latency and
// memory pressure than the first token's speed.
func defaultLoadOptions() loadOptions { return loadOptions{Prefault: "none"} }

func parseLoadOptions(raw string) (loadOptions, error) {
	o := defaultLoadOptions()
	if strings.TrimSpace(raw) == "" {
		return o, nil
	}
	if err := decodeStrictJSON(raw, &o); err != nil {
		return o, fmt.Errorf("invalid load options: %w", err)
	}
	if o.Threads < 0 {
		return o, fmt.Errorf("invalid load options: threads must be >= 0")
	}
	if o.OutOfCore && o.PrepareQuantized {
		return o, fmt.Errorf("invalid load options: out_of_core cannot be combined with prepare_quantized")
	}
	if _, err := prefaultMode(o.Prefault); err != nil {
		return o, fmt.Errorf("invalid load options: %w", err)
	}
	return o, nil
}

// coreOptions converts options parseLoadOptions has already validated.
func (o loadOptions) coreOptions() []gopherllm.Option {
	prefault, _ := prefaultMode(o.Prefault)
	return []gopherllm.Option{
		gopherllm.WithThreads(o.Threads),
		gopherllm.WithPrepareQuantized(o.PrepareQuantized),
		gopherllm.WithOutOfCore(o.OutOfCore),
		gopherllm.WithMmapPrefault(prefault),
		gopherllm.WithMetal(o.Metal),
	}
}

func prefaultMode(value string) (gopherllm.MmapPrefaultMode, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "all":
		return gopherllm.MmapPrefaultAll, nil
	case "core":
		return gopherllm.MmapPrefaultCore, nil
	case "none":
		return gopherllm.MmapPrefaultNone, nil
	default:
		return 0, fmt.Errorf("unsupported prefault mode %q", value)
	}
}

type generationOptions struct {
	MaxTokens     int      `json:"max_tokens"`
	Temperature   float32  `json:"temperature"`
	TopP          float32  `json:"top_p"`
	TopK          int      `json:"top_k"`
	MinP          float32  `json:"min_p"`
	RepeatPenalty float32  `json:"repeat_penalty"`
	Seed          uint64   `json:"seed"`
	SystemPrompt  string   `json:"system_prompt"`
	Stop          []string `json:"stop"`
	// ContextWindowMode is "full" (default: an over-long history is an
	// error), "recent" (drop the oldest complete turns until it fits) or
	// "autoCompress" (condense old turns first, then drop).
	ContextWindowMode string `json:"context_window_mode"`
	// JSONObject constrains the reply to a single JSON object.
	JSONObject bool `json:"json_object"`
}

func parseGenerationOptions(raw string) (gopherllm.GenerationOptions, error) {
	d := gopherllm.DefaultGenerationOptions()
	o := generationOptions{
		MaxTokens:     d.MaxTokens,
		Temperature:   d.Sampler.Temperature,
		TopP:          d.Sampler.TopP,
		TopK:          d.Sampler.TopK,
		MinP:          d.Sampler.MinP,
		RepeatPenalty: d.Sampler.RepeatPenalty,
		SystemPrompt:  d.SystemPrompt,
	}
	if strings.TrimSpace(raw) != "" {
		if err := decodeStrictJSON(raw, &o); err != nil {
			return d, fmt.Errorf("invalid generation options: %w", err)
		}
	}
	d.MaxTokens = o.MaxTokens
	d.Sampler.Temperature = o.Temperature
	d.Sampler.TopP = o.TopP
	d.Sampler.TopK = o.TopK
	d.Sampler.MinP = o.MinP
	d.Sampler.RepeatPenalty = o.RepeatPenalty
	d.Seed = o.Seed
	d.SystemPrompt = o.SystemPrompt
	d.StopSequences = o.Stop
	d.ContextWindowMode = gopherllm.ContextWindowMode(o.ContextWindowMode)
	d.JSONObject = o.JSONObject
	if err := d.Validate(); err != nil {
		return d, fmt.Errorf("invalid generation options: %w", err)
	}
	return d, nil
}

func decodeStrictJSON(raw string, target any) error {
	d := json.NewDecoder(bytes.NewBufferString(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(target); err != nil {
		return err
	}
	if err := d.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			return fmt.Errorf("expected one JSON value")
		}
		return err
	}
	return nil
}
