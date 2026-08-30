package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/rag"
)

// ragState owns the server's document knowledge base: one process-wide
// rag.Index shared by every request, plus the pieces NewHandler wires around
// it that rag.Index itself does not know about — a directory to reseed from
// on demand, a snapshot path to survive a restart, and (when configured) the
// dedicated embedding-model runner backing the index's vector half, which
// this type closes alongside the index it serves. rag.Index is already safe
// for concurrent Add/Search/List/Remove (see its own doc comment); saveMu
// additionally serializes this type's own snapshot writes, since two
// concurrent mutations racing on the same temporary file would otherwise be
// possible.
type ragState struct {
	index *rag.Index

	docsDir      string
	snapshotPath string
	logw         io.Writer

	// embedRunner is the dedicated embedding-model runner backing index's
	// vector half, present only when HandlerOptions.RAGEmbedModelPath was
	// configured and loaded successfully. Owned here (rather than by the
	// caller) so Close releases its memory-mapped weights exactly once,
	// the same lifecycle every other server-owned Runner gets.
	embedRunner *gopherllm.Runner

	saveMu sync.Mutex
}

// ragStateConfig collects newRAGState's inputs. Kept as a struct rather than
// a long positional parameter list because most callers outside tests only
// ever set Options, and the rest are independently optional.
type ragStateConfig struct {
	// Options configures the underlying rag.Index, most notably Embedder —
	// see HandlerOptions.RAGEmbedModelPath for how NewHandler resolves one.
	Options rag.Options
	// DocsDir is the directory seedFromDir indexes at startup and reload
	// re-scans; see HandlerOptions.RAGDocsDir.
	DocsDir string
	// SnapshotPath, if set, is where the index is saved after every mutation
	// and loaded from at startup; see HandlerOptions.RAGSnapshotPath.
	SnapshotPath string
	// LogWriter receives non-fatal diagnostics (a snapshot that failed to
	// save, a reseed that hit an unreadable file). Defaults to io.Discard.
	LogWriter io.Writer
	// EmbedRunner, when non-nil, is the Runner backing Options.Embedder;
	// ragState.close releases it. Nil when no embedding model is configured,
	// or when Options.Embedder is a caller-owned value the caller intends to
	// close itself (a library user calling newRAGState indirectly through
	// NewHandler never sees this field).
	EmbedRunner *gopherllm.Runner
}

// newRAGState builds the index backing Features.RAG. It is always
// constructed (even when the feature is off) so NewHandler has one code path
// regardless of configuration; registerRAGRoutes is what actually decides
// whether anything reaches it from the network.
func newRAGState(cfg ragStateConfig) *ragState {
	logw := cfg.LogWriter
	if logw == nil {
		logw = io.Discard
	}
	return &ragState{
		index:        rag.New(cfg.Options),
		docsDir:      strings.TrimSpace(cfg.DocsDir),
		snapshotPath: strings.TrimSpace(cfg.SnapshotPath),
		logw:         logw,
		embedRunner:  cfg.EmbedRunner,
	}
}

// close releases the dedicated embedding-model runner, if one was loaded.
// The index itself owns no OS resources and needs no closing.
func (s *ragState) close() error {
	if s == nil || s.embedRunner == nil {
		return nil
	}
	return s.embedRunner.Close()
}

// loadSnapshot restores the index from SnapshotPath, if that file exists.
// Called once at startup, before seedFromDir: a document pasted, uploaded, or
// fetched by URL lives only in the snapshot, so it must be in place before
// the directory reseed runs its own Add calls (which replace-by-ID rather
// than merge, so ordering here does not risk losing either source — a
// directory file and a snapshot entry can only collide if they share an ID,
// and a directory-sourced Doc.ID is always its file path).
//
// A missing file is the ordinary first-run case, not an error. A present but
// unreadable or version-mismatched file is logged and skipped rather than
// failing startup — the same fail-open posture seedFromDir already takes for
// a bad --rag-docs path, since an operator's typo or a stale snapshot should
// not stop the server from answering requests with an empty knowledge base.
func (s *ragState) loadSnapshot(opts rag.Options) {
	if s.snapshotPath == "" {
		return
	}
	f, err := os.Open(s.snapshotPath)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			fmt.Fprintf(s.logw, "Warning: rag: load snapshot %s: %v (continuing with an empty knowledge base)\n", s.snapshotPath, err)
		}
		return
	}
	defer f.Close()
	loaded, err := rag.Load(f, opts)
	if err != nil {
		fmt.Fprintf(s.logw, "Warning: rag: load snapshot %s: %v (continuing with an empty knowledge base)\n", s.snapshotPath, err)
		return
	}
	s.index = loaded
}

// save writes the current index to SnapshotPath, replacing it atomically (a
// temp file plus rename, mirroring chatHistoryStore's persistence) so a crash
// mid-write never leaves a truncated snapshot behind. A no-op when no
// SnapshotPath is configured. Errors are logged rather than propagated to the
// HTTP caller: the mutation that triggered this save already succeeded in
// memory and is usable for the rest of this process's life, so a disk problem
// here should not be reported as the add/remove itself having failed.
func (s *ragState) save() {
	if s.snapshotPath == "" {
		return
	}
	s.saveMu.Lock()
	defer s.saveMu.Unlock()
	if err := s.saveLocked(); err != nil {
		fmt.Fprintf(s.logw, "Warning: rag: save snapshot %s: %v\n", s.snapshotPath, err)
	}
}

func (s *ragState) saveLocked() error {
	dir := filepath.Dir(s.snapshotPath)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("create snapshot directory: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".gopherllm-rag-*.tmp")
	if err != nil {
		return fmt.Errorf("create temporary file: %w", err)
	}
	tmpName := tmp.Name()
	removeTemp := true
	defer func() {
		_ = tmp.Close()
		if removeTemp {
			_ = os.Remove(tmpName)
		}
	}()
	if err := tmp.Chmod(0o600); err != nil {
		return fmt.Errorf("protect temporary file: %w", err)
	}
	if err := s.index.Save(tmp); err != nil {
		return fmt.Errorf("encode snapshot: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("sync snapshot: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close snapshot: %w", err)
	}
	if err := os.Rename(tmpName, s.snapshotPath); err != nil {
		return fmt.Errorf("replace snapshot: %w", err)
	}
	removeTemp = false
	return os.Chmod(s.snapshotPath, 0o600)
}

// seedFromDir indexes every plain-text file under dir, using
// rag.Index.AddFS's default extension list and 4 MiB per-file cap — the same
// ingestion policy agent.Agent's WithDocuments applies, reused here so the
// CLI's --rag-docs and the agent package's document loading behave
// identically for the same directory. Called once at startup and again by
// POST /rag/reload to pick up files added or changed since. Errors are
// returned rather than swallowed: a typo'd path should fail startup, not
// silently serve an empty knowledge base.
func (s *ragState) seedFromDir(ctx context.Context, dir string) (added, skipped int, err error) {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return 0, 0, nil
	}
	if _, err := os.Stat(dir); err != nil {
		return 0, 0, fmt.Errorf("rag: seed documents from %s: %w", dir, err)
	}
	return s.index.AddFS(ctx, os.DirFS(dir), rag.IngestOptions{})
}

// newDocID generates an ID for a document the caller did not name — a form
// submission, upload, or URL fetch without one. 16 random bytes hex-encoded
// keeps a pasted or re-pasted document from colliding with a prior one by
// accident, which would silently replace it (see rag.Index.Add's re-add-
// replaces contract) rather than adding a second entry.
func newDocID() string {
	var buf [16]byte
	if _, err := rand.Read(buf[:]); err != nil {
		// crypto/rand failing is effectively unrecoverable for this process;
		// a document ID does not need to be cryptographically unpredictable,
		// only unlikely to collide, so time-based text at least keeps the
		// server answering instead of panicking.
		return fmt.Sprintf("doc-%d-%d", os.Getpid(), time.Now().UnixNano())
	}
	return "doc-" + hex.EncodeToString(buf[:])
}
