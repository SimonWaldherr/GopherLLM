package gopherllm

import "strings"

func (r *Runner) isStopToken(token uint32) bool {
	if r.arch == "gpt-oss" {
		return token == r.tok.EOSID || token == 200002 || token == 200007
	}
	if r.chatTemplateKind() == "kimi-chat" {
		// Kimi K2 declares [EOS] as its tokenizer EOS, but its instruct
		// generation configuration ends an assistant turn with <|im_end|>.
		if id, ok := r.tok.SpecialID("<|im_end|>"); ok && token == id {
			return true
		}
	}
	if qwen35Family(r.arch) {
		// Qwen3.5/3.6/3.8 opens assistant turns with ChatML and terminates them
		// with <|im_end|>. Its tokenizer EOS is not consistently configured to
		// that turn marker across converted GGUFs.
		if id, ok := r.tok.SpecialID("<|im_end|>"); ok && token == id {
			return true
		}
	}
	if r.arch == "exaone4" {
		if id, ok := r.tok.SpecialID("[|endofturn|]"); ok && token == id {
			return true
		}
	}
	if r.arch == "phi3" {
		// Phi-3(.1)-mini's own template ends a turn with <|end|>, and Phi-4's
		// with <|im_end|>; neither is guaranteed to be the tokenizer's
		// declared EOS (observed drifting to <|endoftext|> instead on some
		// GGUF conversions). Checking only the fallback EOSID below let
		// generation run past the model's real stop point into hallucinated
		// extra turns / unrelated text.
		if id, ok := r.tok.SpecialID("<|end|>"); ok && token == id {
			return true
		}
		if id, ok := r.tok.SpecialID("<|im_end|>"); ok && token == id {
			return true
		}
		if id, ok := r.tok.SpecialID("<|endoftext|>"); ok && token == id {
			return true
		}
	}
	if r.arch == "stablelm" {
		// StableLM's ChatML-style template ends a turn with <|im_end|>, which
		// some conversions do not set as the declared EOS, letting the raw
		// tag leak into the visible response text.
		if id, ok := r.tok.SpecialID("<|im_end|>"); ok && token == id {
			return true
		}
	}
	if gemmaFamily(r.arch) {
		if r.arch == "gemma4" {
			// Gemma 4's native turn delimiter replaces the older
			// <end_of_turn> marker used by Gemma 1--3.
			if id, ok := r.tok.SpecialID("<turn|>"); ok && token == id {
				return true
			}
		}
		// Gemma instruct models end assistant turns with <end_of_turn>, not
		// the <eos> the GGUF declares as EOS.
		if id, ok := r.tok.SpecialID("<end_of_turn>"); ok && token == id {
			return true
		}
	}
	return token == r.tok.EOSID
}

func (r *Runner) chatTemplateKind() string {
	if r == nil || r.tok == nil {
		return ""
	}
	if r.gguf != nil {
		if v, ok := r.gguf.Metadata["tokenizer.chat_template"]; ok {
			if s, ok := v.AsString(); ok {
				switch {
				case r.arch == "exaone4" && strings.Contains(s, "[|system|]") && strings.Contains(s, "[|assistant|]") && strings.Contains(s, "[|endofturn|]"):
					return "exaone4-chat"
				case strings.Contains(s, "<|turn>") && strings.Contains(s, "<turn|>"):
					return "gemma4-chat"
				case strings.Contains(s, "[INST]") && strings.Contains(s, "[/INST]"):
					return "mistral-inst"
				case r.arch == "llama" && strings.Contains(s, "<|python_tag|>") && strings.Contains(s, "<|eom_id|>") && strings.Contains(s, "tools_in_user_message"):
					// This is Meta's Llama-3.1-Instruct template, which has a
					// native custom-function / ipython exchange on top of the
					// otherwise shared Llama header protocol.
					return "llama31-chat"
				case strings.Contains(s, "<|start_header_id|>") && strings.Contains(s, "<|eot_id|>"):
					return "header-chat"
				case strings.Contains(s, "<|im_user|>") && strings.Contains(s, "<|im_assistant|>") && strings.Contains(s, "<|im_middle|>") && strings.Contains(s, "<|im_end|>"):
					return "kimi-chat"
				case r.arch == "phi3" && strings.Contains(s, "<|im_start|>") && strings.Contains(s, "<|im_sep|>") && strings.Contains(s, "<|im_end|>"):
					return "phi4-chat"
				case strings.Contains(s, "<|im_start|>") && strings.Contains(s, "<|im_end|>"):
					return "chatml"
				case strings.Contains(s, "<|user|>") && strings.Contains(s, "<|assistant|>") && strings.Contains(s, "<|end|>"):
					return "phi-chat"
				case strings.Contains(s, "<｜User｜>") && strings.Contains(s, "<｜Assistant｜>"):
					return "deepseek-r1-qwen"
				case strings.Contains(s, "<|start_of_role|>") && strings.Contains(s, "<|end_of_role|>"):
					return "granite-chat"
				case strings.Contains(s, "<start_of_turn>") && strings.Contains(s, "<end_of_turn>"):
					return "gemma-chat"
				case strings.Contains(s, "### User:") && strings.Contains(s, "### Assistant:"):
					// Upstage SOLAR's instruct template, inherited by every
					// SOLAR fine-tune (SauerkrautLM-SOLAR-Instruct among them)
					// and shared by a wide class of older Alpaca/Orca-style
					// community checkpoints. It uses no special tokens at all,
					// which is exactly why it needs detecting: without this
					// case it falls through to renderPlainMessages, whose
					// "User: "/"Assistant: " markers are close enough to look
					// plausible and different enough to be out of distribution.
					return "alpaca-chat"
				}
			}
		}
	}
	if _, ok := r.tok.SpecialID("<|im_user|>"); ok {
		if _, ok := r.tok.SpecialID("<|im_assistant|>"); ok {
			if _, ok := r.tok.SpecialID("<|im_middle|>"); ok {
				if _, ok := r.tok.SpecialID("<|im_end|>"); ok {
					return "kimi-chat"
				}
			}
		}
	}
	if r.arch == "exaone4" {
		if _, ok := r.tok.SpecialID("[|system|]"); ok {
			if _, ok := r.tok.SpecialID("[|assistant|]"); ok {
				if _, ok := r.tok.SpecialID("[|endofturn|]"); ok {
					return "exaone4-chat"
				}
			}
		}
	}
	if _, ok := r.tok.SpecialID("[INST]"); ok {
		if _, ok := r.tok.SpecialID("[/INST]"); ok {
			return "mistral-inst"
		}
	}
	if _, ok := r.tok.SpecialID("<|turn>"); ok {
		if _, ok := r.tok.SpecialID("<turn|>"); ok {
			return "gemma4-chat"
		}
	}
	if r.arch == "phi3" {
		if _, ok := r.tok.SpecialID("<|im_start|>"); ok {
			if _, ok := r.tok.SpecialID("<|im_sep|>"); ok {
				if _, ok := r.tok.SpecialID("<|im_end|>"); ok {
					return "phi4-chat"
				}
			}
		}
	}
	if _, ok := r.tok.SpecialID("<|im_start|>"); ok {
		if _, ok := r.tok.SpecialID("<|im_end|>"); ok {
			return "chatml"
		}
	}
	if _, ok := r.tok.SpecialID("<|user|>"); ok {
		if _, ok := r.tok.SpecialID("<|assistant|>"); ok {
			if _, ok := r.tok.SpecialID("<|end|>"); ok {
				return "phi-chat"
			}
		}
	}
	if _, ok := r.tok.SpecialID("<｜User｜>"); ok {
		if _, ok := r.tok.SpecialID("<｜Assistant｜>"); ok {
			return "deepseek-r1-qwen"
		}
	}
	if _, ok := r.tok.SpecialID("<|start_of_role|>"); ok {
		if _, ok := r.tok.SpecialID("<|end_of_role|>"); ok {
			return "granite-chat"
		}
	}
	if _, ok := r.tok.SpecialID("<start_of_turn>"); ok {
		if _, ok := r.tok.SpecialID("<end_of_turn>"); ok {
			return "gemma-chat"
		}
	}
	return ""
}

func specialOr(t *Tokenizer, token string, fallback uint32) uint32 {
	if id, ok := t.SpecialID(token); ok {
		return id
	}
	return fallback
}

func specialOrEncoded(t *Tokenizer, token string) uint32 {
	if id, ok := t.SpecialID(token); ok {
		return id
	}
	ids := t.EncodeWithoutBOS(token)
	if len(ids) > 0 {
		return ids[0]
	}
	return 0
}
