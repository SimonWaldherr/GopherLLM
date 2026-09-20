package gopherllm

import "fmt"

// renderCodestralFIM renders Mistral's Codestral fill-in-the-middle (FIM)
// protocol:
//
//	<s>[PREFIX]{prefix}[SUFFIX]{suffix}[MIDDLE]
//
// Generation continues from [MIDDLE] with no closing tag, the same
// open-ended continuation renderMistralInstMessages uses for a trailing
// assistant prefill. [PREFIX]/[SUFFIX]/[MIDDLE] are Codestral's dedicated
// infill control tokens (present in every real Codestral tokenizer
// alongside [INST]/[SYSTEM_PROMPT]/[TOOL_CALLS], following the same
// bracket-literal convention). A checkpoint without them cannot do FIM
// completion at all, so a missing token is a hard error rather than a
// silent plain-text fallback that would just produce a plausible-looking
// but semantically wrong completion.
func (r *Runner) renderCodestralFIM(prefix, suffix string) ([]uint32, error) {
	prefixTok, ok1 := r.tok.SpecialID("[PREFIX]")
	suffixTok, ok2 := r.tok.SpecialID("[SUFFIX]")
	middleTok, ok3 := r.tok.SpecialID("[MIDDLE]")
	if !(ok1 && ok2 && ok3) {
		return nil, fmt.Errorf("this model's vocabulary has no [PREFIX]/[SUFFIX]/[MIDDLE] tokens: fill-in-the-middle completion requires a Codestral-family checkpoint")
	}
	tokens := make([]uint32, 0, 4+len(prefix)/3+len(suffix)/3)
	if r.tok.AddBOS {
		tokens = append(tokens, r.tok.BOSID)
	}
	tokens = append(tokens, prefixTok)
	tokens = append(tokens, r.tok.EncodeWithoutBOS(prefix)...)
	tokens = append(tokens, suffixTok)
	tokens = append(tokens, r.tok.EncodeWithoutBOS(suffix)...)
	tokens = append(tokens, middleTok)
	return tokens, nil
}
