package gopherllm

import (
	"encoding/json"
	"fmt"
	"strings"
)

// maxAgenticIterations bounds the server-side skill-resolution loop so a
// model that keeps calling load_skill can't spin forever.
const maxAgenticIterations = 6

// RunAgenticChat runs a chat generation, automatically resolving any
// load_skill call the model makes: it looks up the named skill's full body
// and feeds it back as a tool result, then lets the model continue, before
// ever returning to the caller. A tool call for anything else — i.e. every
// tool the CALLER supplied, as opposed to the server's own load_skill — is
// left untouched in the result for the caller to execute and continue via a
// follow-up request with a ToolResultMessage, exactly like ordinary
// (non-agentic) tool use. A turn that mixes a skill call with a caller tool
// call is treated as needing the caller (not resolved internally), so the
// caller never has calls silently dropped out from under it.
//
// A turn's raw output isn't known to be "the final answer" versus a tool call
// until generation for that turn completes, so whenever ANY tool activity is
// possible this turn (skills configured, or the caller supplied tools),
// onToken only fires once, with the complete, already-classified content of
// the winning turn — never with raw, mid-formation tool-call syntax. This
// also holds for plain (non-skill) tool use, since a client streaming
// "[TOOL_CALLS]get_weather[ARGS]..." as if it were visible answer text would
// be exactly the kind of leak this is meant to prevent. When there is no tool
// activity at all for this request, this is a zero-overhead passthrough to
// GenerateChatStreamUntil with full incremental streaming — the common case
// is unaffected.
func RunAgenticChat(r *Runner, messages []ChatMessage, options GenerationOptions, skills []Skill, onToken func(string) bool) (GenerationResult, error) {
	offerSkills := len(skills) > 0 && options.ToolChoice != "none"
	activeTools := options.activeTools()
	if !offerSkills && len(activeTools) == 0 {
		return r.GenerateChatStreamUntil(messages, options, onToken)
	}
	loopOptions := options
	if offerSkills {
		loopOptions.Tools = append(append([]ToolDefinition{}, activeTools...), skillsToolDefinition(skills))
	} else {
		loopOptions.Tools = activeTools
	}

	convo := append([]ChatMessage(nil), messages...)
	var stats GenerationStats
	var result GenerationResult
	var err error
	for range maxAgenticIterations {
		result, err = r.GenerateChatStreamUntil(convo, loopOptions, func(string) bool { return true })
		stats = sumGenerationStats(stats, result.Stats)
		if err != nil {
			break
		}
		resolved, ok := resolveSkillCalls(result.ToolCalls, skills)
		if !ok {
			break
		}
		convo = append(convo, resolved...)
	}
	result.Stats = stats
	if onToken != nil && result.Text != "" {
		onToken(result.Text)
	}
	return result, err
}

// resolveSkillCalls builds the assistant-tool_calls + tool-result messages
// that resolve calls internally, if and only if EVERY call in calls is a
// load_skill invocation (the tool's function name, not the requested skill's
// name, which lives inside its arguments). A call naming an unknown skill
// still gets resolved — with an error result listing the available names —
// so the model can self-correct instead of that leaking to the external
// caller as an unresolvable internal tool call. A turn that contains even one
// call to anything OTHER than load_skill is left entirely to the caller, so a
// mix of a skill call and a real external tool call is never partially
// resolved out from under it. Pure and model-independent, so it is unit
// tested directly against synthetic ToolCall/Skill values.
func resolveSkillCalls(calls []ToolCall, skills []Skill) (resolved []ChatMessage, ok bool) {
	if len(calls) == 0 {
		return nil, false
	}
	for _, c := range calls {
		if c.Function.Name != LoadSkillToolName {
			return nil, false
		}
	}
	resolved = make([]ChatMessage, 0, len(calls)+1)
	resolved = append(resolved, ChatMessage{Role: ChatRoleAssistant, ToolCalls: calls})
	for _, c := range calls {
		resolved = append(resolved, ToolResultMessage(c.ID, c.Function.Name, loadSkillResultContent(c, skills)))
	}
	return resolved, true
}

// loadSkillResultContent produces the tool-result text for one load_skill
// call: the skill's full body on success, or a self-correction hint (valid
// JSON, a parse failure, or a request for an unknown name) on failure.
func loadSkillResultContent(call ToolCall, skills []Skill) string {
	var args struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return fmt.Sprintf("Error: could not parse load_skill arguments as JSON: %v", err)
	}
	if skill, found := findSkill(skills, args.Name); found {
		return skill.Body
	}
	names := make([]string, len(skills))
	for i, s := range skills {
		names[i] = s.Name
	}
	return fmt.Sprintf("Error: no skill named %q. Available skills: %s", args.Name, strings.Join(names, ", "))
}

func sumGenerationStats(a, b GenerationStats) GenerationStats {
	return GenerationStats{
		PromptTokens:    a.PromptTokens + b.PromptTokens,
		GeneratedTokens: a.GeneratedTokens + b.GeneratedTokens,
		PrefillTime:     a.PrefillTime + b.PrefillTime,
		DecodeTime:      a.DecodeTime + b.DecodeTime,
		TotalTime:       a.TotalTime + b.TotalTime,
	}
}
