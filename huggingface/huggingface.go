// Package huggingface is the public Hugging Face Hub client for GopherLLM:
// searching across model tasks and formats, inspecting repository artifacts,
// downloading exact file sets, and resolving GGUF variants for local inference.
//
// It exists as its own package for a specific reason. The Hub client needs
// net/http, which pulls in crypto/tls and the whole HTTP/2 stack, and the root
// inference package deliberately keeps that out of its dependency graph — that
// property is enforced by TestInferencePackageStaysFreeOfServerDependencies
// and is the reason embedding the runtime stays cheap. Importing this package
// is therefore an explicit opt-in to network capability: a program that only
// loads local models never pays for it.
//
// The implementation lives in internal/huggingface so its helpers stay free to
// change. These wrappers are the stable surface, and they are thin by design so
// the CLI, the bundled server, and third-party callers all share one
// implementation and cannot drift apart in caching, offline behaviour, or
// gated-repository handling.
//
// Nothing here contacts the Hub until called. See gopherllm.WritePrivacyReport
// for the full outbound inventory.
package huggingface

import (
	"context"
	"io"

	internalhf "github.com/SimonWaldherr/GopherLLM/internal/huggingface"
)

// Options tunes Hub access. The zero value is online with no progress
// reporting, which is what most callers want.
type Options = internalhf.Options

// Progress reports incremental download progress for one file. Done is true
// exactly once per file; Err is set only on failure. It may be called from
// several goroutines at once when shards download concurrently.
type Progress = internalhf.ProgressEvent

// ProgressFunc receives Progress events.
type ProgressFunc = internalhf.ProgressFunc

// SearchResult is one repository returned by Search.
type SearchResult = internalhf.SearchResult

// RepositoryInfo lists the GGUF variants a repository publishes.
type RepositoryInfo = internalhf.RepositoryInfo

// RepositoryVariant is one downloadable GGUF within a repository.
type RepositoryVariant = internalhf.RepositoryVariant

// DefaultOptions returns the options the CLI uses.
func DefaultOptions() Options { return internalhf.DefaultOptions() }

// Resolve downloads (or reuses from cache) the GGUF named by ref and returns
// its local path, ready for gopherllm.Open.
//
// ref accepts the same forms the CLI does: "owner/repo",
// "owner/repo:file.gguf" to pick one variant of a multi-quant repository, and
// an optional "hf:" prefix. A repository publishing several quantizations
// without an explicit file selector is reported as an ambiguity rather than
// guessed at; use Variants to present the choice.
//
// A sharded repository downloads every shard and returns the first, which is
// all the loader needs — it discovers its siblings. Open such a model with
// gopherllm.WithOutOfCore(true) to keep the shards memory-mapped instead of
// merged into one buffer.
//
// logw receives human-readable progress; pass io.Discard for silence, or set
// Options.OnProgress for structured events.
func Resolve(ctx context.Context, ref string, logw io.Writer, opts Options) (string, error) {
	if logw == nil {
		logw = io.Discard
	}
	return internalhf.ResolveHuggingFaceModelContextWithOptions(ctx, ref, logw, opts)
}

// ResolveFiles is Resolve but returns every local file the reference resolved
// to, in shard order — useful to report the on-disk footprint or to relocate
// the files.
func ResolveFiles(ctx context.Context, ref string, logw io.Writer, opts Options) ([]string, error) {
	if logw == nil {
		logw = io.Discard
	}
	return internalhf.ResolveHuggingFaceModelFilesContextWithOptions(ctx, ref, logw, opts)
}

// Search finds Hub repositories publishing GGUF files. Zero limit uses the
// default; other limits must be in the supported range.
func Search(ctx context.Context, query string, limit int, opts Options) ([]SearchResult, error) {
	return internalhf.SearchGGUFRepositories(ctx, query, limit, opts)
}

// Variants lists the GGUF files a repository publishes, so a caller can offer
// the quantization choice instead of hitting Resolve's ambiguity error.
func Variants(ctx context.Context, ref string, opts Options) (RepositoryInfo, error) {
	return internalhf.RepositoryVariants(ctx, ref, opts)
}

// List writes a human-readable listing of a repository's GGUF variants.
func List(ctx context.Context, ref string, out io.Writer, opts Options) error {
	return internalhf.ListGGUFContextWithOptions(ctx, ref, out, opts)
}

// Repository normalises a reference to its "owner/repo" form.
func Repository(ref string) (string, error) { return internalhf.Repository(ref) }

// SearchOptions selects model task and format independently.
type SearchOptions = internalhf.SearchOptions

// File describes any repository artifact, including weights and companion files.
type File = internalhf.File

// Manifest describes the available files at a repository revision.
type Manifest = internalhf.Manifest

// SearchModels searches all model types. Empty task/format filters include all.
// Metadata is not an assertion of local runtime support.
func SearchModels(ctx context.Context, search SearchOptions, opts Options) ([]SearchResult, error) {
	return internalhf.SearchModels(ctx, search, opts)
}

// Inspect lists all repository artifacts without downloading model weights.
func Inspect(ctx context.Context, ref string, opts Options) (Manifest, error) {
	return internalhf.Inspect(ctx, ref, opts)
}

// DownloadFiles downloads exact repository paths and preserves subdirectories.
// It does not execute code, select a runtime, or infer a complete model bundle.
func DownloadFiles(ctx context.Context, ref string, files []string, logw io.Writer, opts Options) ([]string, error) {
	return internalhf.DownloadFiles(ctx, ref, files, logw, opts)
}
