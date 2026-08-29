package agent

import (
	"context"
	"fmt"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/rag"
)

// Option configures an Agent at construction time. Options are applied in
// order, but every option that touches the retrieval index (WithDocuments,
// WithIndex, WithIndexFile, WithEmbedder, WithSelfEmbedding, and the Chunker/
// Ranking-affecting ones layered on top of rag.Options) is order-independent:
// each just records what was asked for, and Open/New builds or loads the
// actual *rag.Index once, after every Option has run. A failing option (a
// duplicate tool name, an unreadable document path) fails Open/New outright;
// nothing is partially applied.
type Option func(ctx context.Context, a *Agent) error

// pendingIndex accumulates every index-affecting Option before Open/New
// resolves them into an actual *rag.Index. Kept on Agent itself (as
// unexported fields, listed here for one place to reason about the whole
// group) rather than a separate struct, since Agent is already the value
// threaded through every Option.
type pendingIndexFields struct {
	providedIndex *rag.Index
	docsToLearn   []string
	ragOptions    rag.Options
	selfEmbed     bool
}

// resolveIndex turns the accumulated WithDocuments/WithIndex/WithIndexFile/
// WithEmbedder/WithSelfEmbedding options into a.index, called once after
// every Option has run.
func (a *Agent) resolveIndex(ctx context.Context) error {
	// WithIndex takes the provided index exactly as given and returns
	// early: WithDocuments/WithIndexFile/WithEmbedder/WithSelfEmbedding are
	// all decisions about how to BUILD an index, which a caller handing over
	// a ready one has already made themselves.
	if a.pending.providedIndex != nil {
		a.index = a.pending.providedIndex
		return nil
	}

	if a.pending.selfEmbed {
		a.pending.ragOptions.Embedder = a.model
		if a.pending.ragOptions.EmbedderID == "" {
			a.pending.ragOptions.EmbedderID = "self"
		}
	}
	if a.indexPath != "" {
		ix, err := loadIndexFile(a.indexPath, a.pending.ragOptions)
		if err != nil {
			return err
		}
		a.index = ix
	} else {
		a.index = rag.New(a.pending.ragOptions)
	}

	if len(a.pending.docsToLearn) > 0 {
		if err := a.Learn(ctx, a.pending.docsToLearn...); err != nil {
			return err
		}
	}
	return nil
}

// WithDocuments queues files and directories to be indexed once the Agent's
// index is resolved (after every Option has run — see Learn for directory
// vs. file handling). Combine with WithIndexFile to skip re-reading these on
// every process start once a snapshot exists... except this package does not
// yet detect "unchanged since last save": a document already present under
// the same ID is simply re-chunked and re-embedded, which is correct but not
// incrementally fast. Point WithIndexFile at a stable path and accept the
// re-ingest cost, or call Agent.Learn yourself only for what actually
// changed, if that cost matters for your corpus size.
func WithDocuments(paths ...string) Option {
	return func(ctx context.Context, a *Agent) error {
		a.pending.docsToLearn = append(a.pending.docsToLearn, paths...)
		return nil
	}
}

// WithIndex uses ix directly as the Agent's index, taking precedence over
// WithIndexFile/WithEmbedder/WithSelfEmbedding/WithDocuments (which are
// ignored when this is set — a caller handing over a ready index has already
// made those decisions).
func WithIndex(ix *rag.Index) Option {
	return func(ctx context.Context, a *Agent) error {
		a.pending.providedIndex = ix
		return nil
	}
}

// WithIndexFile loads the snapshot at path if it exists (on Open/New) and
// writes it back to the same path on Close. Ignored if WithIndex is also
// used.
func WithIndexFile(path string) Option {
	return func(ctx context.Context, a *Agent) error {
		a.indexPath = path
		return nil
	}
}

// WithTool registers tools on the Agent; see ToolSet.Add for the
// construction-time checks.
func WithTool(tools ...Tool) Option {
	return func(ctx context.Context, a *Agent) error {
		return a.tools.Add(tools...)
	}
}

// WithSkills offers skills through the same load_skill mechanism
// gopherllm.RunAgenticChat already provides.
func WithSkills(skills []gopherllm.Skill) Option {
	return func(ctx context.Context, a *Agent) error {
		a.skills = append(a.skills, skills...)
		return nil
	}
}

// WithSkillsDir loads every SKILL.md under dir via gopherllm.LoadSkills.
func WithSkillsDir(dir string) Option {
	return func(ctx context.Context, a *Agent) error {
		skills, err := gopherllm.LoadSkills(dir)
		if err != nil {
			return fmt.Errorf("agent: load skills from %s: %w", dir, err)
		}
		a.skills = append(a.skills, skills...)
		return nil
	}
}

// WithQuery sets the rag.Query used for every retrieval this Agent performs,
// in either RetrievalMode.
func WithQuery(q rag.Query) Option {
	return func(ctx context.Context, a *Agent) error {
		a.query = q
		return nil
	}
}

// WithChunker sets the Chunker used when this Agent indexes documents.
func WithChunker(c rag.Chunker) Option {
	return func(ctx context.Context, a *Agent) error {
		a.pending.ragOptions.Chunker = c
		return nil
	}
}

// WithRanking sets the scoring weights used when this Agent's index ranks
// search results.
func WithRanking(r rag.Ranking) Option {
	return func(ctx context.Context, a *Agent) error {
		a.pending.ragOptions.Ranking = r
		return nil
	}
}

// WithMaxToolRounds bounds how many tool-result rounds a Chat call will run
// before forcing a final, tools-withdrawn pass. Zero (the default) leaves
// gopherllm.GenerationOptions.MaxToolRounds's own default in effect.
func WithMaxToolRounds(n int) Option {
	return func(ctx context.Context, a *Agent) error {
		a.maxRounds = n
		return nil
	}
}

// WithStepObserver receives every Step of every Chat call as it happens,
// synchronously — see gopherllm.AgentObserver, which this wraps.
func WithStepObserver(fn func(Step)) Option {
	return func(ctx context.Context, a *Agent) error {
		a.observer = fn
		return nil
	}
}

// WithRetrieval sets how Chat uses the index. The default is RetrieveByTool.
func WithRetrieval(mode RetrievalMode) Option {
	return func(ctx context.Context, a *Agent) error {
		a.retrieval = mode
		return nil
	}
}

// WithEmbedder turns on the vector half of retrieval using a second,
// dedicated encoder model (BGE/E5/Nomic/Granite) rather than the Agent's own
// chat model. See rag.Options.Embedder for the full reasoning why this is not
// the default.
func WithEmbedder(e rag.Embedder, embedderID string) Option {
	return func(ctx context.Context, a *Agent) error {
		a.pending.ragOptions.Embedder = e
		a.pending.ragOptions.EmbedderID = embedderID
		return nil
	}
}

// WithSelfEmbedding uses the Agent's own chat model as the embedder instead
// of a dedicated encoder. This is the option that costs the most, and is
// explained here rather than three files away in gopherllm.Runner.Embed's
// doc comment, because this is the option a reader is about to call:
//
//   - Every chunk you index becomes a full prefill pass through the CHAT
//     model AND clears its KV prefix cache (the decoder embedding path does
//     this on every call), so the next chat turn after Learn re-prefills
//     from scratch instead of reusing anything.
//   - The chat model's mean-pooled hidden states were never trained as an
//     embedding head. On a small local model they measurably lose to plain
//     BM25 at exactly the queries people type at their own documents: part
//     numbers, error codes, invoice ids, surnames.
//
// It exists for the case where accepting both costs is still better than a
// second multi-gigabyte download. If you can afford a second model, prefer
// WithEmbedder with a dedicated BGE/E5/Nomic GGUF.
func WithSelfEmbedding() Option {
	return func(ctx context.Context, a *Agent) error {
		a.pending.selfEmbed = true
		return nil
	}
}
