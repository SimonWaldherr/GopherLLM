// Package mobile is the deliberately small, gomobile-friendly public surface
// for using GopherLLM from Swift and Objective-C.
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

func (o loadOptions) coreOptions() ([]gopherllm.Option, error) {
	prefault, err := prefaultMode(o.Prefault)
	if err != nil {
		return nil, err
	}
	return []gopherllm.Option{
		gopherllm.WithThreads(o.Threads),
		gopherllm.WithPrepareQuantized(o.PrepareQuantized),
		gopherllm.WithOutOfCore(o.OutOfCore),
		gopherllm.WithMmapPrefault(prefault),
		gopherllm.WithMetal(o.Metal),
	}, nil
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
}

func defaultGenerationOptions() generationOptions {
	o := gopherllm.DefaultGenerationOptions()
	return generationOptions{MaxTokens: o.MaxTokens, Temperature: o.Sampler.Temperature, TopP: o.Sampler.TopP, TopK: o.Sampler.TopK, MinP: o.Sampler.MinP, RepeatPenalty: o.Sampler.RepeatPenalty, SystemPrompt: o.SystemPrompt}
}

func parseGenerationOptions(raw string) (generationOptions, error) {
	o := defaultGenerationOptions()
	if strings.TrimSpace(raw) != "" {
		if err := decodeStrictJSON(raw, &o); err != nil {
			return o, fmt.Errorf("invalid generation options: %w", err)
		}
	}
	core := gopherllm.ApplyGenOptions(nil, o.coreOptions()...)
	if err := core.Validate(); err != nil {
		return o, fmt.Errorf("invalid generation options: %w", err)
	}
	return o, nil
}

func (o generationOptions) coreOptions() []gopherllm.GenOption {
	return []gopherllm.GenOption{gopherllm.WithMaxTokens(o.MaxTokens), gopherllm.WithTemperature(o.Temperature), gopherllm.WithTopP(o.TopP), gopherllm.WithTopK(o.TopK), gopherllm.WithMinP(o.MinP), gopherllm.WithRepeatPenalty(o.RepeatPenalty), gopherllm.WithSeed(o.Seed), gopherllm.WithSystemPrompt(o.SystemPrompt), gopherllm.WithStop(o.Stop...)}
}

func decodeStrictJSON(raw string, target any) error {
	d := json.NewDecoder(bytes.NewBufferString(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(target); err != nil {
		return err
	}
	if err := d.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			return fmt.Errorf("expected one JSON object")
		}
		return err
	}
	return nil
}
