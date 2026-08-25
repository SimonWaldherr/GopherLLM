package server

import (
	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func applyRequestOptions(def gopherllm.GenerationOptions, maxTokens *int, temp *float32, topP *float32, topK *int, minP *float32, repeat *float32, seed *uint64, system *string, stop any, tools []gopherllm.ToolDefinition, toolChoice string) gopherllm.GenerationOptions {
	o := def
	if maxTokens != nil {
		o.MaxTokens = *maxTokens
	}
	if temp != nil {
		o.Sampler.Temperature = *temp
	}
	if topP != nil {
		o.Sampler.TopP = *topP
	}
	if topK != nil {
		o.Sampler.TopK = *topK
	}
	if minP != nil {
		o.Sampler.MinP = *minP
	}
	if repeat != nil {
		o.Sampler.RepeatPenalty = *repeat
	}
	if seed != nil {
		o.Seed = *seed
	}
	if system != nil {
		o.SystemPrompt = *system
	}
	if parsed, ok := parseStop(stop); ok {
		o.StopSequences = parsed
	}
	if len(tools) > 0 {
		o.Tools = tools
	}
	if toolChoice != "" {
		o.ToolChoice = toolChoice
	}
	return o
}

func firstFloat(a, b *float32) *float32 {
	if a != nil {
		return a
	}
	return b
}

func parseStop(v any) ([]string, bool) {
	switch x := v.(type) {
	case string:
		return []string{x}, true
	case []any:
		out := []string{}
		for _, item := range x {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out, true
	case nil:
		return nil, false
	default:
		return nil, false
	}
}

func firstStop(a, b any) any {
	if a != nil {
		return a
	}
	return b
}
