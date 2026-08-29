// Package agent composes a gopherllm.Model, a rag.Index and a set of tools
// into Ask/Chat with citations — the layer that used to be two hundred lines
// of everybody's main.go.
//
// It adds no new inference or agent-loop code: Ask and Chat call
// gopherllm.RunAgenticChatWithGenerator against the composed Model, tools and
// skills. Every hardening behavior (per-call timeout, panic recovery, result
// truncation, provenance framing, cross-round caching, self-correcting
// unknown tool names) is the root package's, applied uniformly to every tool
// here exactly as it is to gopherllm.RunAgenticChatWithTools callers
// elsewhere in the codebase — there is exactly one agent loop.
//
// Tool execution in this design is strictly sequential — there is no
// parallel-tool-execution option. SearchDocumentsTool's request-scoped
// context.Value collector (see retrieval.go) relies on this: adding
// parallelism later means revisiting that mechanism, not just flipping a
// flag.
package agent

import (
	"context"
	"fmt"
	"os"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/rag"
)

// RetrievalMode selects how an Agent uses its index during Chat.
type RetrievalMode int

const (
	// RetrieveByTool offers search_documents as a tool and lets the model
	// decide whether to search at all. The default.
	RetrieveByTool RetrievalMode = iota
	// RetrievePrefix always searches once, using the conversation's most
	// recent user message as the query, and prepends the results to the
	// system prompt.
	RetrievePrefix
	// RetrieveOff never touches the index during Chat, even if one is
	// configured — useful for a conversation turn that should answer from
	// the model alone (see Answer.Retrieved for detecting which happened).
	RetrieveOff
)

// Agent is a Model plus an optional rag.Index and tool set.
type Agent struct {
	model     *gopherllm.Model
	ownsModel bool
	// generate is what Chat actually calls: a.model.Runner().GenerateChatStreamUntil
	// for every real Agent, or a scripted gopherllm.ChatGenerator for a test
	// built via newAgentWithGenerator. Keeping this as a field (rather than
	// always deriving it from a.model) is what lets Chat's whole
	// retrieval/tool/citation composition be tested without a GGUF, the same
	// reason the root package exports ChatGenerator/RunAgenticChatWithGenerator.
	generate  gopherllm.ChatGenerator
	index     *rag.Index
	indexPath string
	tools     *ToolSet
	skills    []gopherllm.Skill
	retrieval RetrievalMode
	query     rag.Query
	maxRounds int
	observer  func(Step)

	// pending accumulates every index-affecting Option (WithDocuments,
	// WithIndex, WithEmbedder, WithSelfEmbedding, WithChunker, WithRanking)
	// until resolveIndex builds or loads the real *rag.Index once, after
	// every Option has run. See options.go.
	pending pendingIndexFields
}

// Open loads a GGUF and returns a ready Agent; Close releases the model.
// Options are applied in order and any failure (a duplicate tool name, an
// unreadable document path) fails Open rather than being silently dropped,
// so a misconfigured agent never reaches production answering questions with
// half its tools.
func Open(ctx context.Context, modelPath string, opts ...Option) (*Agent, error) {
	m, err := gopherllm.Open(ctx, modelPath)
	if err != nil {
		return nil, err
	}
	a, err := newAgent(ctx, m, true, opts)
	if err != nil {
		m.Close()
		return nil, err
	}
	return a, nil
}

// New wraps a Model the caller already owns. Close does not close it.
func New(ctx context.Context, m *gopherllm.Model, opts ...Option) (*Agent, error) {
	return newAgent(ctx, m, false, opts)
}

func newAgent(ctx context.Context, m *gopherllm.Model, ownsModel bool, opts []Option) (*Agent, error) {
	return buildAgent(ctx, m, m.Runner().GenerateChatStreamUntil, ownsModel, opts)
}

// newAgentWithGenerator builds an Agent around a scripted generator instead
// of a real Model, for testing Chat's retrieval/tool/citation composition
// without a GGUF. Not exported: an external caller who needs to test their
// own tools against the real loop should use
// gopherllm.RunAgenticChatWithGenerator directly, which is the seam this one
// is built on.
func newAgentWithGenerator(ctx context.Context, generate gopherllm.ChatGenerator, opts []Option) (*Agent, error) {
	return buildAgent(ctx, nil, generate, false, opts)
}

func buildAgent(ctx context.Context, m *gopherllm.Model, generate gopherllm.ChatGenerator, ownsModel bool, opts []Option) (*Agent, error) {
	a := &Agent{model: m, generate: generate, ownsModel: ownsModel, tools: &ToolSet{names: map[string]bool{}}}
	for _, opt := range opts {
		if err := opt(ctx, a); err != nil {
			return nil, err
		}
	}
	if err := a.resolveIndex(ctx); err != nil {
		return nil, err
	}
	return a, nil
}

// Close releases the Agent. If WithIndexFile was used, the index is saved
// back to that path first; a save failure is returned even though the model
// (if this Agent owns it) is still closed, since a lost index is a real
// problem for the caller to know about, not something to swallow on the way
// out.
func (a *Agent) Close() error {
	var saveErr error
	if a.indexPath != "" {
		saveErr = saveIndexFile(a.index, a.indexPath)
	}
	if a.ownsModel {
		if err := a.model.Close(); err != nil {
			return err
		}
	}
	return saveErr
}

// Model returns the underlying Model.
func (a *Agent) Model() *gopherllm.Model { return a.model }

// Index returns the underlying retrieval index (never nil).
func (a *Agent) Index() *rag.Index { return a.index }

// Tools returns the names of every registered tool, in registration order.
func (a *Agent) Tools() []string { return a.tools.Names() }

// Learn chunks and indexes files and directories. A directory is walked with
// rag.IngestOptions defaults (see rag.Index.AddFS); a single file is read and
// indexed directly, using its base name as the Doc title.
func (a *Agent) Learn(ctx context.Context, paths ...string) error {
	for _, p := range paths {
		info, err := os.Stat(p)
		if err != nil {
			return fmt.Errorf("agent: learn %s: %w", p, err)
		}
		if info.IsDir() {
			if _, _, err := a.index.AddFS(ctx, os.DirFS(p), rag.IngestOptions{}); err != nil {
				return fmt.Errorf("agent: learn %s: %w", p, err)
			}
			continue
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return fmt.Errorf("agent: learn %s: %w", p, err)
		}
		if err := a.index.Add(ctx, rag.Doc{ID: p, Title: baseName(p), Path: p, Text: string(data)}); err != nil {
			return fmt.Errorf("agent: learn %s: %w", p, err)
		}
	}
	return nil
}

// AddTool registers tools; see ToolSet.Add for the construction-time checks
// (a duplicate or reserved name, an empty Execute, is rejected here rather
// than discovered mid-conversation).
func (a *Agent) AddTool(tools ...Tool) error { return a.tools.Add(tools...) }

// Ask is Chat over a single user message.
func (a *Agent) Ask(ctx context.Context, question string, opts ...gopherllm.GenOption) (Answer, error) {
	return a.Chat(ctx, []gopherllm.ChatMessage{gopherllm.UserMessage(question)}, opts...)
}

// Chat answers messages using the index and registered tools as needed,
// per the Agent's RetrievalMode.
func (a *Agent) Chat(ctx context.Context, messages []gopherllm.ChatMessage, opts ...gopherllm.GenOption) (Answer, error) {
	options := a.buildOptions(opts)
	tools := a.tools.AgenticTools()

	searched := false
	var sources []Source
	switch {
	case a.retrieval == RetrieveByTool && a.index.Len() > 0:
		tools = append(tools, gopherllm.AgenticTool(SearchDocumentsTool(a.index, a.query)))
		ctx = withSourceCollector(ctx, &sources)
	case a.retrieval == RetrievePrefix && a.index.Len() > 0:
		hits, err := a.index.Search(ctx, lastUserText(messages), a.query)
		if err == nil && len(hits) > 0 {
			for _, h := range hits {
				sources = append(sources, sourceFromHit(h))
			}
			options.SystemPrompt = prependContext(options.SystemPrompt, hits)
			searched = true
		}
	}
	options = options.WithContext(ctx)

	var steps []Step
	observe := func(e gopherllm.AgentEvent) {
		s := stepFromEvent(e)
		if s.Tool == SearchDocumentsToolName && s.Kind == StepToolStart {
			searched = true
		}
		steps = append(steps, s)
		if a.observer != nil {
			a.observer(s)
		}
	}

	result, err := gopherllm.RunAgenticChatWithGenerator(a.generate,
		messages, options, a.skills, tools, func(string) bool { return true }, observe)
	if err != nil {
		return Answer{}, err
	}
	return Answer{
		Text: result.Text, Reasoning: result.ReasoningText,
		Sources: dedupeSources(sources), Steps: steps,
		Retrieved: searched, FinishReason: result.FinishReason, Stats: result.Stats,
	}, nil
}

func (a *Agent) buildOptions(opts []gopherllm.GenOption) gopherllm.GenerationOptions {
	options := gopherllm.DefaultGenerationOptions()
	for _, opt := range opts {
		opt(&options)
	}
	if a.maxRounds > 0 {
		options.MaxToolRounds = a.maxRounds
	}
	return options
}

func baseName(p string) string {
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] == '/' || p[i] == '\\' {
			return p[i+1:]
		}
	}
	return p
}

// Answer is one completed Chat turn.
type Answer struct {
	Text      string
	Reasoning string
	// Sources is deduplicated (by DocID+ChunkIndex, highest score kept), best
	// first, and contains only what a tool or RetrievePrefix search actually
	// returned.
	Sources []Source
	Steps   []Step
	// Retrieved reports whether the index was actually queried this turn.
	// Meaningful mainly for RetrieveByTool (the default) with a non-empty
	// index, where the model decides whether to search at all: False is not
	// itself a problem — a question needing no documents should get exactly
	// this — but it is the one fact an empty Sources cannot distinguish,
	// "answered from the index, which came back thin" from "the model never
	// looked."
	Retrieved    bool
	FinishReason string
	Stats        gopherllm.GenerationStats
}

// Source is one citation: a chunk a search actually returned this turn.
type Source struct {
	DocID, Title, Path string
	ChunkIndex         int
	Start, End         int
	Score              float32
	Excerpt            string
}

// StepKind names one step of an Agent.Chat turn.
type StepKind string

const (
	StepModelStart StepKind = "model_start"
	StepModelEnd   StepKind = "model_end"
	StepToolStart  StepKind = "tool_start"
	StepToolEnd    StepKind = "tool_end"
)

// Step is Agent's view of one gopherllm.AgentEvent, renamed and reshaped for
// this package's audience rather than passing the root package's wire type
// straight through.
type Step struct {
	Kind                     StepKind
	Round                    int
	Tool                     string
	CallID                   string
	Arguments, Result, Error string
	Cached, Truncated        bool
	Duration                 time.Duration
}

func stepFromEvent(e gopherllm.AgentEvent) Step {
	kind := StepToolStart
	switch e.Kind {
	case gopherllm.AgentEventToolEnd:
		kind = StepToolEnd
	case gopherllm.AgentEventIteration:
		kind = StepModelStart
	}
	return Step{
		Kind: kind, Round: e.Iteration, Tool: e.Tool, CallID: e.CallID,
		Arguments: e.Arguments, Result: e.Result, Error: e.Error,
		Cached: e.Cached, Truncated: e.Truncated, Duration: e.Duration,
	}
}

func saveIndexFile(ix *rag.Index, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("agent: save index %s: %w", path, err)
	}
	defer f.Close()
	if err := ix.Save(f); err != nil {
		return fmt.Errorf("agent: save index %s: %w", path, err)
	}
	return nil
}

func loadIndexFile(path string, opts rag.Options) (*rag.Index, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return rag.New(opts), nil
		}
		return nil, fmt.Errorf("agent: load index %s: %w", path, err)
	}
	defer f.Close()
	ix, err := rag.Load(f, opts)
	if err != nil {
		return nil, fmt.Errorf("agent: load index %s: %w", path, err)
	}
	return ix, nil
}
