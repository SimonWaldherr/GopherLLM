package gopherllm

import (
	"encoding/json"
	"fmt"
	"strings"
)

// renderMistralInstMessages renders the Mistral/Ministral instruct format:
//
//	<s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][AVAILABLE_TOOLS]{tools}[/AVAILABLE_TOOLS][INST]{user}[/INST]{assistant}</s>...
//
// [INST]/[/INST] (and, on newer Tekken vocabularies, [SYSTEM_PROMPT]/
// [/SYSTEM_PROMPT] and the tool-calling markers) are emitted as control
// tokens. When the vocabulary lacks the dedicated system-prompt tokens we
// fall back to the older Mistral 2410 behavior of folding the system prompt
// into the final user turn. Format verified directly against the
// tokenizer.chat_template of a real Ministral-3-3B-Instruct-2512 GGUF.
// renderMistralInstMessages renders the [INST]/[/INST] Mistral-family
// template. Its bool return follows the same "not applicable, try another
// renderer" convention every renderMessages branch uses (true only for the
// rare case of a vocabulary missing [INST]/[/INST]) — its error return is a
// distinct signal: ok=true with a non-nil error means the template DID
// apply but rendering a message's attached image failed (too many images,
// no vision projector loaded, bad image bytes). Callers must treat that as
// a real failure, not silently fall through to a different renderer, since
// no other renderer understands ChatMessage.Images and would just drop the
// image data instead of erroring.
func (r *Runner) renderMistralInstMessages(messages []ChatMessage, systemPrompt string, tools []ToolDefinition) ([]uint32, map[int][]float32, bool, error) {
	// This renderer is also used by PrepareChatContext, which deliberately
	// does not take genLock. Keep the whole image-rendering transaction under
	// one shared tower lease so Close cannot release Metal/WebGPU resources or
	// unmap a borrowed projector while EncodeImagePixtral is using it.
	r.visionMu.RLock()
	defer r.visionMu.RUnlock()
	instTok, ok1 := r.tok.SpecialID("[INST]")
	instEndTok, ok2 := r.tok.SpecialID("[/INST]")
	if !(ok1 && ok2) {
		return nil, nil, false, nil
	}
	sysStart, sysEnd, hasSysTokens := r.systemPromptTokens()
	callTok := r.mistralMarker("[TOOL_CALLS]")
	argsTok := r.mistralMarker("[ARGS]")
	resultsStart := r.mistralMarker("[TOOL_RESULTS]")
	resultsEnd := r.mistralMarker("[/TOOL_RESULTS]")

	system := strings.TrimSpace(systemPrompt)
	lastUser := -1
	lastNonSystem := -1
	for i, m := range messages {
		if m.Role == ChatRoleSystem {
			if s := strings.TrimSpace(m.Content); s != "" {
				system = s
			}
			continue
		}
		lastNonSystem = i
		if m.Role == ChatRoleUser {
			lastUser = i
		}
	}
	toolsJSON := ""
	if len(tools) > 0 {
		// Keep the historical behavior for a malformed tool definition: omit
		// the unavailable-tool block rather than failing an otherwise usable
		// chat request. The exact serialized form is also the render-cache key,
		// so mutations to a caller-owned definition cannot return stale tokens.
		if encoded, err := json.Marshal(tools); err == nil {
			toolsJSON = string(encoded)
		}
	}

	tokenCap := 128 + len(messages)*16
	if hasSysTokens {
		tokenCap += 12
	}
	if len(system) > 0 {
		tokenCap += len(system) / 4
	}
	if toolsJSON != "" {
		tokenCap += 32 + len(toolsJSON)/4
	}
	tokens := make([]uint32, 0, tokenCap)

	// This cache key covers only text emitted before every chat turn. It is
	// therefore safe to use even when a later user message has image content:
	// neither image placeholder positions nor image embeddings enter the cache.
	cacheKey := mistralPromptPrefixCacheKey{toolsJSON: toolsJSON}
	if hasSysTokens {
		cacheKey.system = system
	}
	var cacheEpoch uint64
	if cached, epoch, hit := r.appendMistralPromptPrefix(tokens, cacheKey); hit {
		tokens = cached
	} else {
		cacheEpoch = epoch
		prefixStart := len(tokens)
		if r.tok.AddBOS {
			tokens = append(tokens, r.tok.BOSID)
		}
		if system != "" && hasSysTokens {
			tokens = append(tokens, sysStart)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(system)...)
			tokens = append(tokens, sysEnd)
		}
		if toolsJSON != "" {
			tokens = append(tokens, r.mistralMarker("[AVAILABLE_TOOLS]")...)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(toolsJSON)...)
			tokens = append(tokens, r.mistralMarker("[/AVAILABLE_TOOLS]")...)
		}
		r.putMistralPromptPrefix(cacheKey, tokens[prefixStart:], cacheEpoch)
	}

	var imageEmbeds map[int][]float32
	for idx, m := range messages {
		switch m.Role {
		case ChatRoleSystem:
			continue
		case ChatRoleAssistant:
			if content := strings.TrimSpace(m.Content); content != "" {
				tokens = append(tokens, r.tok.EncodeWithoutBOS(content)...)
			}
			for _, call := range m.ToolCalls {
				args := call.Function.Arguments
				if args == "" {
					args = "{}"
				}
				tokens = append(tokens, callTok...)
				tokens = append(tokens, r.tok.EncodeWithoutBOS(call.Function.Name)...)
				tokens = append(tokens, argsTok...)
				tokens = append(tokens, r.tok.EncodeWithoutBOS(args)...)
			}
			// A trailing assistant message is a prefill continuation: leave
			// the turn open (no EOS) so generation extends it — the standard
			// way to seed a reply prefix with Mistral/Devstral models.
			if idx == lastNonSystem && len(m.ToolCalls) == 0 {
				break
			}
			tokens = append(tokens, r.tok.EOSID)
		case ChatRoleTool:
			tokens = append(tokens, resultsStart...)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(m.Content)...)
			tokens = append(tokens, resultsEnd...)
		default:
			content := strings.TrimSpace(m.Content)
			if idx == lastUser && system != "" && !hasSysTokens {
				content = system + "\n\n" + content
			}
			tokens = append(tokens, instTok)
			if len(m.Images) > 0 {
				if len(m.Images) > 1 {
					return nil, nil, true, fmt.Errorf("rendering message %d: only one image per message is supported, got %d", idx, len(m.Images))
				}
				// We already hold visionMu; calling HasVision here would take a
				// recursive RLock and can deadlock when Close is waiting for its
				// exclusive lease.
				if r.vision == nil {
					return nil, nil, true, fmt.Errorf("rendering message %d: message includes an image but no vision projector is loaded for this model", idx)
				}
				imgTok, ok1 := r.tok.SpecialID("[IMG]")
				breakTok, ok2 := r.tok.SpecialID("[IMG_BREAK]")
				endTok, ok3 := r.tok.SpecialID("[IMG_END]")
				if !(ok1 && ok2 && ok3) {
					return nil, nil, true, fmt.Errorf("rendering message %d: this model's vocabulary is missing the [IMG]/[IMG_BREAK]/[IMG_END] special tokens image content requires", idx)
				}
				embeds, mergedRows, mergedCols, err := r.encodeChatImage(m.Images[0])
				if err != nil {
					return nil, nil, true, fmt.Errorf("rendering message %d: %w", idx, err)
				}
				if imageEmbeds == nil {
					imageEmbeds = make(map[int][]float32, len(embeds)+mergedRows)
				}
				for row := 0; row < mergedRows; row++ {
					for col := 0; col < mergedCols; col++ {
						imageEmbeds[len(tokens)] = embeds[row*mergedCols+col]
						tokens = append(tokens, imgTok)
					}
					if row < mergedRows-1 {
						imageEmbeds[len(tokens)] = r.vision.ImgBreak
						tokens = append(tokens, breakTok)
					}
				}
				tokens = append(tokens, endTok)
			}
			tokens = append(tokens, r.tok.EncodeWithoutBOS(content)...)
			tokens = append(tokens, instEndTok)
		}
	}
	return tokens, imageEmbeds, true, nil
}

// mistralStaticPromptPrefix returns the exact token IDs that the Mistral
// renderer emits before the first dynamic chat turn: BOS, an optional native
// system block, and optional available-tools block. It deliberately reuses the
// renderer itself (with only system-role messages) instead of duplicating its
// whitespace, JSON, and special-token rules in the KV-cache path.
//
// A static K/V snapshot is useful only when it is a proper prefix of the
// already-rendered prompt. The final prefix check makes this a conservative
// no-op for legacy Mistral templates that fold a system prompt into a user
// turn, malformed tool definitions, and any future template variation which
// changes the relative layout.
func (r *Runner) mistralStaticPromptPrefix(messages []ChatMessage, systemPrompt string, tools []ToolDefinition, fullTokens []uint32) []uint32 {
	if r == nil || r.kind != loadedStandard || r.chatTemplateKind() != "mistral-inst" || len(fullTokens) <= 1 {
		return nil
	}
	var systems []ChatMessage
	for _, message := range messages {
		if message.Role == ChatRoleSystem {
			systems = append(systems, message)
		}
	}
	prefix, _, ok, err := r.renderMistralInstMessages(systems, systemPrompt, tools)
	if err != nil || !ok || len(prefix) <= 1 || len(prefix) >= len(fullTokens) {
		return nil
	}
	if sharedTokenPrefix(prefix, fullTokens) != len(prefix) {
		return nil
	}
	return prefix
}

// mistralMarker returns literal as a single control token when the vocabulary
// defines one (true for every marker on real Mistral/Ministral Tekken
// tokenizers, verified directly against a Ministral-3-3B-Instruct-2512 GGUF),
// falling back to plain-text encoding so rendering degrades rather than fails
// outright on a hypothetical vocabulary that lacks it.
func (r *Runner) mistralMarker(literal string) []uint32 {
	if id, ok := r.tok.SpecialID(literal); ok {
		return []uint32{id}
	}
	return r.tok.EncodeWithoutBOS(literal)
}

func (r *Runner) systemPromptTokens() (start, end uint32, ok bool) {
	s, ok1 := r.tok.SpecialID("[SYSTEM_PROMPT]")
	e, ok2 := r.tok.SpecialID("[/SYSTEM_PROMPT]")
	return s, e, ok1 && ok2
}
