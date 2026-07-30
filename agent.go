package gopherllm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/SimonWaldherr/GopherLLM/internal/tooling"
)

// ToolCallFunction is the OpenAI-compatible function payload of a tool call.
type ToolCallFunction = tooling.CallFunction

// ToolCall is one function call requested by the assistant, OpenAI-compatible.
type ToolCall = tooling.Call

// ToolFunctionDef describes a callable function, OpenAI-compatible.
type ToolFunctionDef = tooling.FunctionDefinition

// ToolDefinition is one entry of an OpenAI-compatible "tools" array.
type ToolDefinition = tooling.Definition

func newToolCallID(rng *Rng) string { return tooling.NewCallID(rng.NextF32) }

func validToolCallID(id string) bool { return tooling.ValidCallID(id) }

func findTool(tools []ToolDefinition, name string) (ToolFunctionDef, bool) {
	return tooling.Find(tools, name)
}

func toolNames(tools []ToolDefinition) []string { return tooling.Names(tools) }

// maxAgenticIterations bounds the server-side skill-resolution loop so a
// model that keeps calling load_skill can't spin forever.
const maxAgenticIterations = 6

// AgenticTool is a server-owned tool that the agent loop may execute itself.
// Unlike a caller-supplied ToolDefinition, its result is fed straight back to
// the model. This is appropriate for bounded, read-only integrations such as
// knowledge retrieval; caller-owned tools remain the caller's responsibility.
type AgenticTool struct {
	Definition ToolDefinition
	Execute    func(context.Context, ToolCall) (string, error)
}

// AgentEventKind names a step of the agentic loop.
type AgentEventKind string

const (
	// AgentEventToolStart fires just before a tool executes.
	AgentEventToolStart AgentEventKind = "tool_start"
	// AgentEventToolEnd fires when it returns, carrying Duration and either
	// Result or Error.
	AgentEventToolEnd AgentEventKind = "tool_end"
	// AgentEventIteration fires when the model is asked to continue after a
	// round of tool results.
	AgentEventIteration AgentEventKind = "iteration"
)

// AgentEvent reports one step of the agentic loop while it happens.
//
// Without this the loop is a black box: a request that fires three Wikipedia
// lookups and two model passes looks identical, from the outside, to one that
// simply took a long time. Callers use these events to show what ran, with
// what arguments, and how long each part took.
type AgentEvent struct {
	Kind      AgentEventKind `json:"kind"`
	Iteration int            `json:"iteration"`
	Tool      string         `json:"tool,omitempty"`
	// Arguments is the raw JSON the model passed, truncated for display.
	Arguments string `json:"arguments,omitempty"`
	// Result is the tool's output, truncated. Error is set instead when the
	// tool failed; the loop continues either way, feeding the error back to
	// the model so it can correct itself.
	Result   string        `json:"result,omitempty"`
	Error    string        `json:"error,omitempty"`
	Duration time.Duration `json:"-"`
	// DurationMS is the wire-friendly form of Duration.
	DurationMS int64 `json:"duration_ms,omitempty"`
}

// AgentObserver receives loop progress. It is called synchronously from the
// generating goroutine, so it must not block.
type AgentObserver func(AgentEvent)

// agentEventTextLimit keeps a single event small enough to stream cheaply; the
// full tool result still goes to the model, only the report is trimmed.
const agentEventTextLimit = 600

func truncateForEvent(s string) string {
	if len(s) <= agentEventTextLimit {
		return s
	}
	return s[:agentEventTextLimit] + "…"
}

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
	return runAgenticChat(r, messages, options, skills, nil, onToken, nil)
}

// RunAgenticChatWithTools is RunAgenticChat with bounded, server-owned tools.
// It resolves a turn only when every call belongs to the supplied tools or to
// load_skill. A mixed turn is returned untouched so a client never loses one
// of its own tool calls.
func RunAgenticChatWithTools(r *Runner, messages []ChatMessage, options GenerationOptions, skills []Skill, tools []AgenticTool, onToken func(string) bool) (GenerationResult, error) {
	return runAgenticChat(r, messages, options, skills, tools, onToken, nil)
}

// RunAgenticChatObserved is RunAgenticChatWithTools plus progress reporting.
// observe may be nil, in which case this is exactly RunAgenticChatWithTools.
func RunAgenticChatObserved(r *Runner, messages []ChatMessage, options GenerationOptions, skills []Skill, tools []AgenticTool, onToken func(string) bool, observe AgentObserver) (GenerationResult, error) {
	return runAgenticChat(r, messages, options, skills, tools, onToken, observe)
}

// chatGenerator matches Runner.GenerateChatStreamUntil's signature. Threading
// it through as a value (rather than calling r directly) lets tests replace a
// real model with a scripted one, which is the only practical way to pin down
// loop behaviour like the iteration cap below — a real GGUF model can't be
// made to reliably keep requesting tools for a fixed number of rounds.
type chatGenerator func([]ChatMessage, GenerationOptions, func(string) bool) (GenerationResult, error)

func runAgenticChat(r *Runner, messages []ChatMessage, options GenerationOptions, skills []Skill, tools []AgenticTool, onToken func(string) bool, observe AgentObserver) (GenerationResult, error) {
	return runAgenticChatWith(r.GenerateChatStreamUntil, messages, options, skills, tools, onToken, observe)
}

func runAgenticChatWith(generate chatGenerator, messages []ChatMessage, options GenerationOptions, skills []Skill, tools []AgenticTool, onToken func(string) bool, observe AgentObserver) (GenerationResult, error) {
	loopOptions, agentic := AgenticOptionsForTools(options, skills, tools)
	if !agentic {
		return generate(messages, options, onToken)
	}

	convo := append([]ChatMessage(nil), messages...)
	var stats GenerationStats
	var result GenerationResult
	var err error
	exhausted := false
	for iteration := 1; iteration <= maxAgenticIterations; iteration++ {
		if observe != nil && iteration > 1 {
			observe(AgentEvent{Kind: AgentEventIteration, Iteration: iteration})
		}
		result, err = generate(convo, loopOptions, func(string) bool { return true })
		stats = sumGenerationStats(stats, result.Stats)
		if err != nil {
			break
		}
		resolved, ok := resolveInternalToolCalls(options.generationContext(), result.ToolCalls, skills, tools, iteration, observe)
		if !ok {
			break
		}
		convo = append(convo, resolved...)
		exhausted = iteration == maxAgenticIterations
	}
	// The iteration budget ran out right after a tool call was resolved: the
	// model never got to see that last result, so `result` is still its
	// pre-resolution tool_calls request. Returning that as-is would hand the
	// caller an empty, unactionable "answer" (finish_reason tool_calls, no
	// text). Force one more pass with tools withdrawn so the model must
	// summarize in words — grounded by the same instruction that already
	// stops it from inventing what those tools didn't return.
	if exhausted {
		if observe != nil {
			observe(AgentEvent{Kind: AgentEventIteration, Iteration: maxAgenticIterations + 1})
		}
		finalOptions := loopOptions
		finalOptions.Tools = nil
		finalOptions.ToolChoice = "none"
		finalResult, finalErr := generate(convo, finalOptions, func(string) bool { return true })
		stats = sumGenerationStats(stats, finalResult.Stats)
		if finalErr == nil {
			result = finalResult
		} else if err == nil {
			err = finalErr
		}
	}
	result.Stats = stats
	if onToken != nil && result.Text != "" {
		onToken(result.Text)
	}
	return result, err
}

// AgenticOptionsFor returns the effective generation settings for the next
// agent loop iteration. Keeping this separate lets the HTTP handler measure a
// recent-context request against the same tool definition that the model will
// actually see before it starts an SSE response.
func AgenticOptionsFor(options GenerationOptions, skills []Skill) (GenerationOptions, bool) {
	return AgenticOptionsForTools(options, skills, nil)
}

// groundingInstruction is appended to the system prompt whenever tools are
// actually active this turn. Without it, a tool result that under-answers the
// question (a one-sentence lead extract instead of the list the user asked
// for) leaves the model to pattern-complete the gap with plausible-sounding
// invented specifics — names, dates, places that were never in any tool
// output. The loop already streams every tool call and result back to the
// caller (see AgentObserver) so a fabrication is visible in hindsight; this
// instruction is the attempt to stop it happening in the first place.
const groundingInstruction = "When you have used a tool, answer only from the facts its results actually contain. " +
	"If a result is incomplete or does not cover what was asked (for example, a summary that states a count without listing the items), say plainly what is missing instead of inventing specific names, numbers, or dates to fill the gap."

// AgenticOptionsForTools adds server-owned tool definitions before applying
// tool_choice, so a request can opt into a built-in integration without the
// browser needing to send executable tool schemas itself.
func AgenticOptionsForTools(options GenerationOptions, skills []Skill, tools []AgenticTool) (GenerationOptions, bool) {
	if len(tools) > 0 {
		options.Tools = append(append([]ToolDefinition{}, options.Tools...), agenticToolDefinitions(tools)...)
	}
	offerSkills := len(skills) > 0 && options.ToolChoice != "none"
	activeTools := options.ActiveTools()
	if !offerSkills && len(activeTools) == 0 {
		return options, false
	}
	loopOptions := options
	if offerSkills {
		loopOptions.Tools = append(append([]ToolDefinition{}, activeTools...), skillsToolDefinition(skills))
	} else {
		loopOptions.Tools = activeTools
	}
	if strings.TrimSpace(loopOptions.SystemPrompt) == "" {
		loopOptions.SystemPrompt = groundingInstruction
	} else {
		loopOptions.SystemPrompt = strings.TrimRight(loopOptions.SystemPrompt, " \t\n") + "\n\n" + groundingInstruction
	}
	return loopOptions, true
}

func agenticToolDefinitions(tools []AgenticTool) []ToolDefinition {
	defs := make([]ToolDefinition, 0, len(tools))
	for _, tool := range tools {
		if tool.Execute != nil && tool.Definition.Type == "function" && tool.Definition.Function.Name != "" {
			defs = append(defs, tool.Definition)
		}
	}
	return defs
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
	return resolveInternalToolCalls(context.Background(), calls, skills, nil, 1, nil)
}

func resolveInternalToolCalls(ctx context.Context, calls []ToolCall, skills []Skill, tools []AgenticTool, iteration int, observe AgentObserver) (resolved []ChatMessage, ok bool) {
	if len(calls) == 0 {
		return nil, false
	}
	byName := make(map[string]AgenticTool, len(tools))
	for _, tool := range tools {
		if tool.Execute != nil && tool.Definition.Function.Name != "" {
			byName[tool.Definition.Function.Name] = tool
		}
	}
	for _, c := range calls {
		if c.Function.Name != LoadSkillToolName && byName[c.Function.Name].Execute == nil {
			return nil, false
		}
	}
	resolved = make([]ChatMessage, 0, len(calls)+1)
	resolved = append(resolved, ChatMessage{Role: ChatRoleAssistant, ToolCalls: calls})
	for _, c := range calls {
		if observe != nil {
			observe(AgentEvent{
				Kind: AgentEventToolStart, Iteration: iteration,
				Tool: c.Function.Name, Arguments: truncateForEvent(c.Function.Arguments),
			})
		}
		started := time.Now()
		content, failure := "", ""
		if c.Function.Name == LoadSkillToolName {
			content = loadSkillResultContent(c, skills)
		} else if tool := byName[c.Function.Name]; tool.Execute != nil {
			result, err := tool.Execute(ctx, c)
			if err != nil {
				// The error goes back to the model as the tool result so it can
				// retry or explain, and is reported separately for display.
				failure = err.Error()
				content = fmt.Sprintf("Error: %s tool failed: %v", c.Function.Name, err)
			} else {
				content = result
			}
		}
		if observe != nil {
			elapsed := time.Since(started)
			event := AgentEvent{
				Kind: AgentEventToolEnd, Iteration: iteration, Tool: c.Function.Name,
				Duration: elapsed, DurationMS: elapsed.Milliseconds(),
			}
			if failure != "" {
				event.Error = truncateForEvent(failure)
			} else {
				event.Result = truncateForEvent(content)
			}
			observe(event)
		}
		resolved = append(resolved, ToolResultMessage(c.ID, c.Function.Name, content))
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
	ttft := a.TTFT
	if ttft == 0 {
		ttft = b.TTFT
	}
	return GenerationStats{
		PromptTokens:    a.PromptTokens + b.PromptTokens,
		GeneratedTokens: a.GeneratedTokens + b.GeneratedTokens,
		TTFT:            ttft,
		PrefillTime:     a.PrefillTime + b.PrefillTime,
		DecodeTime:      a.DecodeTime + b.DecodeTime,
		TotalTime:       a.TotalTime + b.TotalTime,
	}
}
