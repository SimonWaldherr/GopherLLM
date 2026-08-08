package huggingface

// Hugging Face GGUF import.  This deliberately uses the Hub's HTTP API
// directly instead of shelling out to git, Python, or a separate downloader,
// so the CLI remains a single Go binary on every supported platform.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const hfPrefix = "hf:"

type hfReference struct {
	Repository string
	Quant      string
	Revision   string
}

// Options controls Hub operations without relying on a global HTTP transport.
// Offline mirrors Hugging Face's HF_HUB_OFFLINE convention: no Hub request is
// made and only complete snapshots already in the shared cache are considered.
type Options struct {
	Offline bool
	// OnProgress, when set, receives periodic byte-progress updates for each
	// file transferred by a resolve/download call. It is throttled the same
	// way the textual logw progress line is (roughly 4 times per second) plus
	// one call at the start and end of each file, so a caller such as a web
	// UI can drive a progress bar without polling or parsing log text.
	OnProgress ProgressFunc
}

// ProgressEvent reports incremental progress for one file transferred from
// the Hub. Done is true exactly once per file, on the final event (whether it
// succeeded or failed; Err is set only on failure).
type ProgressEvent struct {
	File      string
	Completed int64
	Total     int64
	Done      bool
	Err       error
}

// ProgressFunc receives ProgressEvents. It may be called from goroutines
// downloading different shards concurrently, so implementations that share
// mutable state must synchronize themselves.
type ProgressFunc func(ProgressEvent)

// DefaultOptions reads the Hugging Face-compatible offline setting. Explicit
// Options supplied by a host or the CLI can enable offline mode as well.
func DefaultOptions() Options {
	return Options{Offline: hfBoolEnv("HF_HUB_OFFLINE")}
}

// ParseHuggingFaceReference parses hf:owner/repository[:quant][@revision].
// The quant selector is matched against GGUF filenames, for example
// hf:bartowski/Qwen3-4B-GGUF:Q4_K_M@main.
func ParseHuggingFaceReference(value string) (hfReference, error) {
	if !strings.HasPrefix(strings.ToLower(value), hfPrefix) {
		return hfReference{}, fmt.Errorf("Hugging Face reference must start with %q", hfPrefix)
	}
	v := strings.TrimSpace(value[len(hfPrefix):])
	if v == "" {
		return hfReference{}, fmt.Errorf("Hugging Face repository is empty")
	}
	r := hfReference{Revision: "main"}
	if at := strings.LastIndexByte(v, '@'); at >= 0 {
		r.Revision, v = strings.TrimSpace(v[at+1:]), strings.TrimSpace(v[:at])
		if r.Revision == "" {
			return r, fmt.Errorf("Hugging Face revision is empty")
		}
	}
	if colon := strings.LastIndexByte(v, ':'); colon >= 0 {
		r.Quant, v = strings.TrimSpace(v[colon+1:]), strings.TrimSpace(v[:colon])
		if r.Quant == "" {
			return r, fmt.Errorf("Hugging Face quantization is empty")
		}
	}
	if parts := strings.Split(v, "/"); len(parts) != 2 || parts[0] == "" || parts[1] == "" || strings.ContainsAny(v, "\\?") {
		return r, fmt.Errorf("Hugging Face repository must be owner/repository")
	}
	r.Repository = v
	return r, nil
}

type hfTreeEntry struct {
	Path string `json:"path"`
	Type string `json:"type"`
	Size int64  `json:"size"`
}

type hfCache struct {
	repository string
	root       string
}

type hfDownloadTask struct {
	name string
	path string
	size int64
}

// synchronizedWriter lets concurrent shard workers share a caller-provided
// diagnostic writer without racing or interleaving individual writes.
type synchronizedWriter struct {
	mu sync.Mutex
	w  io.Writer
}

func (w *synchronizedWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.w.Write(p)
}

// GGUFOption is one downloadable model variant in a Hugging Face repository.
// Split files are presented as one option because GopherLLM always downloads
// the complete shard set needed to load them.
type GGUFOption struct {
	File      string
	Quant     string
	SizeBytes int64
	Shards    int
}

type ambiguousGGUFError struct {
	candidates []string
}

func (e *ambiguousGGUFError) Error() string {
	return "multiple GGUF files match: " + strings.Join(e.candidates, ", ")
}

// ListGGUF writes the selectable GGUF variants in an HF repository. ref may
// be either owner/repository[@revision] or a regular hf: reference.
func ListGGUF(ref string, out io.Writer) error {
	return ListGGUFContext(context.Background(), ref, out)
}

// ListGGUFContext is ListGGUF with cancellation support for callers that
// attach the operation to a request, CLI signal, or shutdown context.
func ListGGUFContext(ctx context.Context, ref string, out io.Writer) error {
	return ListGGUFContextWithOptions(ctx, ref, out, DefaultOptions())
}

// ListGGUFContextWithOptions is ListGGUFContext with explicit network policy.
// In offline mode it lists only complete variants from the cached revision.
func ListGGUFContextWithOptions(ctx context.Context, ref string, out io.Writer, opts Options) error {
	r, variants, err := listGGUFVariants(ctx, ref, opts)
	if err != nil {
		return err
	}
	if out == nil {
		out = io.Discard
	}
	fmt.Fprintf(out, "GGUF variants in %s", r.Repository)
	if r.Revision != "main" {
		fmt.Fprintf(out, "@%s", r.Revision)
	}
	fmt.Fprintln(out, ":")
	fmt.Fprintf(out, "%-14s %-10s %-7s %s\n", "quant", "size", "shards", "run")
	for _, option := range variants {
		quant := option.Quant
		if quant == "" {
			quant = "unknown"
		}
		fmt.Fprintf(out, "%-14s %-10s %-7d %s\n", quant, formatHFSize(option.SizeBytes), option.Shards, hfSelector(r.Repository, r.Revision, option))
	}
	return nil
}

// RepositoryVariant is one downloadable GGUF variant together with the exact
// selector ResolveHuggingFaceModelContextWithOptions (or
// ResolveHuggingFaceModelFilesContextWithOptions) accepts to fetch it.
type RepositoryVariant struct {
	Quant     string
	SizeBytes int64
	Shards    int
	Selector  string
}

// RepositoryInfo is the structured result of listing a Hugging Face
// repository's GGUF variants, for callers (such as a web UI) that want to
// render or filter the list themselves instead of the preformatted text
// ListGGUFContextWithOptions produces.
type RepositoryInfo struct {
	Repository string
	Revision   string
	Variants   []RepositoryVariant
}

// RepositoryVariants resolves the selectable GGUF variants in a Hugging Face
// repository reference (accepting the same owner/repository[:quant][@rev]
// forms as ListGGUF, with or without the "hf:" prefix) without downloading
// anything.
func RepositoryVariants(ctx context.Context, ref string, opts Options) (RepositoryInfo, error) {
	r, variants, err := listGGUFVariants(ctx, ref, opts)
	if err != nil {
		return RepositoryInfo{}, err
	}
	info := RepositoryInfo{Repository: r.Repository, Revision: r.Revision, Variants: make([]RepositoryVariant, 0, len(variants))}
	for _, option := range variants {
		quant := option.Quant
		if quant == "" {
			quant = "unknown"
		}
		info.Variants = append(info.Variants, RepositoryVariant{
			Quant:     quant,
			SizeBytes: option.SizeBytes,
			Shards:    option.Shards,
			Selector:  hfSelector(r.Repository, r.Revision, option),
		})
	}
	return info, nil
}

// Repository returns the owner/repository portion of a Hugging Face
// reference, for callers that need to name a destination (for example, a
// local directory) after the source repository without duplicating
// ParseHuggingFaceReference's parsing rules.
func Repository(ref string) (string, error) {
	r, err := ParseHuggingFaceReference(ref)
	if err != nil {
		return "", err
	}
	return r.Repository, nil
}

func listGGUFVariants(ctx context.Context, ref string, opts Options) (hfReference, []GGUFOption, error) {
	if err := ctx.Err(); err != nil {
		return hfReference{}, nil, err
	}
	offline := opts.Offline || hfBoolEnv("HF_HUB_OFFLINE")
	ref = strings.TrimSpace(ref)
	if !strings.HasPrefix(strings.ToLower(ref), hfPrefix) {
		ref = hfPrefix + ref
	}
	r, err := ParseHuggingFaceReference(ref)
	if err != nil {
		return hfReference{}, nil, err
	}
	cache := newHFCache(r)
	var entries []hfTreeEntry
	if offline {
		snapshot := cache.snapshotForRef(r.Revision)
		if snapshot == "" {
			return r, nil, offlineCacheError(r)
		}
		entries, err = cachedHFGGUFEntries(ctx, snapshot)
		if err != nil {
			return r, nil, fmt.Errorf("read cached Hugging Face repository %s: %w", r.Repository, err)
		}
	} else {
		entries, _, err = hfListFiles(ctx, r)
		if err != nil {
			return r, nil, err
		}
	}
	variants := ggufOptions(entries, r.Quant)
	if offline {
		variants = completeCachedGGUFOptions(entries, variants)
	}
	if len(variants) == 0 {
		if offline {
			return r, nil, offlineCacheError(r)
		}
		return r, nil, fmt.Errorf("no GGUF files found in Hugging Face repository %s", r.Repository)
	}
	return r, variants, nil
}

// hfSelector builds the exact "hf:owner/repository[:quant][@revision]"
// string that resolves back to option, for callers that print or return a
// runnable reference alongside a listed variant.
func hfSelector(repository, revision string, option GGUFOption) string {
	selector := hfPrefix + repository
	if option.Quant != "" {
		selector += ":" + option.Quant
	} else {
		selector += ":" + option.File
	}
	if revision != "main" {
		selector += "@" + revision
	}
	return selector
}

// ResolveHuggingFaceModel downloads (or reuses) the GGUF selected by ref and
// returns its local first-shard path. Files use Hugging Face's standard
// blobs/refs/snapshots layout, so Python, JavaScript, and GopherLLM share a
// single on-disk cache rather than downloading the same GGUF repeatedly.
func ResolveHuggingFaceModel(ref string, logw io.Writer) (string, error) {
	return ResolveHuggingFaceModelContext(context.Background(), ref, logw)
}

// ResolveHuggingFaceModelContext resolves and downloads a GGUF while honoring
// ctx. Cancellation stops repository listing, metadata requests, and any
// in-flight file transfer while preserving the partial blob for a later resume.
func ResolveHuggingFaceModelContext(ctx context.Context, ref string, logw io.Writer) (string, error) {
	return ResolveHuggingFaceModelContextWithOptions(ctx, ref, logw, DefaultOptions())
}

// ResolveHuggingFaceModelContextWithOptions resolves a GGUF with an explicit
// network policy. Offline mode never contacts the Hub and requires a complete
// snapshot for the selected revision to be present in the local cache.
func ResolveHuggingFaceModelContextWithOptions(ctx context.Context, ref string, logw io.Writer, options Options) (string, error) {
	files, err := resolveHuggingFaceFiles(ctx, ref, logw, options)
	if err != nil {
		return "", err
	}
	return files[0], nil
}

// ResolveHuggingFaceModelFilesContextWithOptions is
// ResolveHuggingFaceModelContextWithOptions but returns every local shard
// path, in shard order, instead of only the first. Callers that only need a
// loadable model path should prefer ResolveHuggingFaceModelContextWithOptions;
// this variant exists for callers that must place the complete shard set
// somewhere else on disk (for example, linking a split model's files into a
// model directory for local discovery), since a split GGUF cannot be loaded
// from its first shard alone if the siblings are missing.
func ResolveHuggingFaceModelFilesContextWithOptions(ctx context.Context, ref string, logw io.Writer, options Options) ([]string, error) {
	return resolveHuggingFaceFiles(ctx, ref, logw, options)
}

func resolveHuggingFaceFiles(ctx context.Context, ref string, logw io.Writer, options Options) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r, err := ParseHuggingFaceReference(ref)
	if err != nil {
		return nil, err
	}
	if logw == nil {
		logw = io.Discard
	} else if logw != io.Discard {
		logw = &synchronizedWriter{w: logw}
	}
	cache := newHFCache(r)
	if options.Offline || hfBoolEnv("HF_HUB_OFFLINE") {
		if snapshot := cache.snapshotForRef(r.Revision); snapshot != "" {
			if files, err := cachedGGUFFiles(ctx, snapshot, r.Quant); err == nil && len(files) > 0 {
				fmt.Fprintf(logw, "Hugging Face: offline; using cached %s\n", filepath.Base(files[0]))
				return files, nil
			}
		}
		return nil, offlineCacheError(r)
	}
	entries, commit, err := hfListFiles(ctx, r)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		// A previously resolved revision remains usable without a connection.
		if snapshot := cache.snapshotForRef(r.Revision); snapshot != "" {
			if files, cacheErr := cachedGGUFFiles(ctx, snapshot, r.Quant); cacheErr == nil && len(files) > 0 {
				fmt.Fprintf(logw, "Hugging Face: offline; using cached %s\n", filepath.Base(files[0]))
				return files, nil
			}
		}
		return nil, err
	}
	files, err := selectHFGGUF(entries, r.Quant)
	if err != nil {
		var ambiguous *ambiguousGGUFError
		if errors.As(err, &ambiguous) {
			return nil, fmt.Errorf("%w; run `gopherllm --hf-list %s` to choose an exact selector", err, hfListReference(r))
		}
		return nil, err
	}
	if commit == "" {
		// The public Hub currently supplies X-Repo-Commit. Keeping this
		// fallback also makes the importer compatible with simple Hub mirrors.
		commit = safeHFPathPart(r.Revision)
	}
	snapshot := cache.snapshot(commit)
	if cached, err := cachedGGUFFiles(ctx, snapshot, r.Quant); err == nil && len(cached) > 0 {
		if err := cache.writeRef(r.Revision, commit); err != nil {
			return nil, err
		}
		fmt.Fprintf(logw, "Hugging Face: using cached %s\n", filepath.Base(cached[0]))
		return cached, nil
	}
	if err := os.MkdirAll(snapshot, 0o755); err != nil {
		return nil, fmt.Errorf("create Hugging Face cache: %w", err)
	}
	tasks := make([]hfDownloadTask, 0, len(files))
	for _, name := range files {
		local := filepath.Join(snapshot, filepath.FromSlash(name))
		if st, err := os.Stat(local); err == nil && st.Size() > 0 {
			fmt.Fprintf(logw, "Hugging Face: using cached %s\n", name)
			continue
		}
		tasks = append(tasks, hfDownloadTask{name: name, path: local, size: hfFileSize(entries, name)})
	}
	if err := downloadHFFiles(ctx, r, cache, tasks, logw, options.OnProgress); err != nil {
		return nil, err
	}
	if err := cache.writeRef(r.Revision, commit); err != nil {
		return nil, err
	}
	resolved := make([]string, len(files))
	for i, name := range files {
		resolved[i] = filepath.Join(snapshot, filepath.FromSlash(name))
	}
	return resolved, nil
}

// downloadHFFiles downloads at most three independent GGUF shards at once.
// The worker pool makes split-model imports materially faster without opening
// an unbounded number of large transfers. The first failure cancels siblings;
// their .incomplete blobs remain available for resume on the next invocation.
func downloadHFFiles(parent context.Context, r hfReference, cache hfCache, tasks []hfDownloadTask, logw io.Writer, progress ProgressFunc) error {
	if len(tasks) == 0 {
		return nil
	}
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	workers := min(3, len(tasks))
	jobs := make(chan hfDownloadTask, len(tasks))
	for _, task := range tasks {
		jobs <- task
	}
	close(jobs)
	var downloaded atomic.Int64
	var firstErr error
	var errOnce sync.Once
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range jobs {
				if ctx.Err() != nil {
					return
				}
				blob, err := hfDownload(ctx, r, task.name, cache, task.size, logw, progress, func(n int64) { downloaded.Add(n) })
				if err == nil {
					err = linkHFCacheFile(blob, task.path)
				}
				if err != nil {
					errOnce.Do(func() {
						firstErr = err
						cancel()
					})
					return
				}
			}
		}()
	}
	wg.Wait()
	if firstErr != nil {
		return firstErr
	}
	if err := parent.Err(); err != nil {
		return err
	}
	if added := downloaded.Load(); added > 0 && logw != nil && logw != io.Discard {
		fmt.Fprintf(logw, "Hugging Face: downloaded %s in this run\n", formatHFSize(added))
	}
	return nil
}

func hfFileSize(entries []hfTreeEntry, name string) int64 {
	for _, entry := range entries {
		if entry.Path == name {
			return entry.Size
		}
	}
	return 0
}

func hfListReference(r hfReference) string {
	ref := r.Repository
	if r.Quant != "" {
		ref += ":" + r.Quant
	}
	if r.Revision != "main" {
		ref += "@" + r.Revision
	}
	return ref
}

func offlineCacheError(r hfReference) error {
	return fmt.Errorf("Hugging Face offline mode is enabled; no complete cached GGUF matches %s. Connect once without --hf-offline or unset HF_HUB_OFFLINE=1", hfPrefix+hfListReference(r))
}

func hfBoolEnv(name string) bool {
	switch strings.ToUpper(strings.TrimSpace(os.Getenv(name))) {
	case "1", "ON", "YES", "TRUE":
		return true
	default:
		return false
	}
}

func hfEndpoint() string {
	if endpoint := strings.TrimRight(strings.TrimSpace(os.Getenv("HF_ENDPOINT")), "/"); endpoint != "" {
		return endpoint
	}
	return "https://huggingface.co"
}

func newHFCache(r hfReference) hfCache {
	home := strings.TrimSpace(os.Getenv("HF_HOME"))
	if home == "" {
		if userCache, err := os.UserCacheDir(); err == nil {
			home = filepath.Join(userCache, "huggingface")
		} else {
			home = ".cache/huggingface"
		}
	}
	name := "models--" + strings.ReplaceAll(r.Repository, "/", "--")
	return hfCache{repository: r.Repository, root: filepath.Join(home, "hub", name)}
}

func (c hfCache) snapshot(commit string) string {
	return filepath.Join(c.root, "snapshots", safeHFPathPart(commit))
}

func (c hfCache) snapshotForRef(revision string) string {
	ref, err := os.ReadFile(filepath.Join(c.root, "refs", safeHFRefPath(revision)))
	if err != nil {
		return ""
	}
	commit := strings.TrimSpace(string(ref))
	if commit == "" {
		return ""
	}
	return c.snapshot(commit)
}

func (c hfCache) writeRef(revision, commit string) error {
	path := filepath.Join(c.root, "refs", safeHFRefPath(revision))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create Hugging Face ref cache: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".gopherllm-ref-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err := io.WriteString(tmp, commit+"\n"); err == nil {
		err = tmp.Close()
	} else {
		_ = tmp.Close()
	}
	if err != nil {
		return fmt.Errorf("write Hugging Face ref cache: %w", err)
	}
	if err := os.Rename(name, path); err != nil {
		return fmt.Errorf("store Hugging Face ref cache: %w", err)
	}
	return nil
}

func safeHFPathPart(s string) string {
	s = strings.ReplaceAll(s, "/", "--")
	s = strings.ReplaceAll(s, "\\", "--")
	if s == "." || s == "" {
		return "main"
	}
	return s
}

func safeHFRefPath(revision string) string {
	parts := strings.Split(revision, "/")
	for i, part := range parts {
		if part == "" || part == "." || part == ".." {
			return "main"
		}
		parts[i] = safeHFPathPart(part)
	}
	return filepath.Join(parts...)
}

func hfListFiles(ctx context.Context, r hfReference) ([]hfTreeEntry, string, error) {
	u := hfEndpoint() + "/api/models/" + hfRepositoryPath(r.Repository) + "/tree/" + url.PathEscape(r.Revision) + "?recursive=true"
	var entries []hfTreeEntry
	var commit string
	for u != "" {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
		if err != nil {
			return nil, "", err
		}
		hfAuth(req)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, "", fmt.Errorf("list Hugging Face repository %s: %w", r.Repository, err)
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			err := hfStatusError(resp, r.Repository)
			resp.Body.Close()
			return nil, "", err
		}
		if value := strings.TrimSpace(resp.Header.Get("X-Repo-Commit")); value != "" {
			commit = value
		}
		var page []hfTreeEntry
		err = json.NewDecoder(resp.Body).Decode(&page)
		next := hfNextPage(resp.Header.Get("Link"))
		resp.Body.Close()
		if err != nil {
			return nil, "", fmt.Errorf("read Hugging Face repository %s: %w", r.Repository, err)
		}
		entries = append(entries, page...)
		if strings.HasPrefix(next, "/") {
			next = hfEndpoint() + next
		}
		u = next
	}
	return entries, commit, nil
}

func hfNextPage(link string) string {
	for _, value := range strings.Split(link, ",") {
		if !strings.Contains(value, `rel="next"`) && !strings.Contains(value, "rel=next") {
			continue
		}
		start, end := strings.IndexByte(value, '<'), strings.IndexByte(value, '>')
		if start >= 0 && end > start+1 {
			return value[start+1 : end]
		}
	}
	return ""
}

func selectHFGGUF(entries []hfTreeEntry, quant string) ([]string, error) {
	groups := ggufGroups(entries)
	var candidates []string
	for key := range groups {
		if quant == "" || strings.Contains(strings.ToLower(key), strings.ToLower(quant)) {
			candidates = append(candidates, key)
		}
	}
	sort.Strings(candidates)
	if len(candidates) == 0 {
		if quant != "" {
			return nil, fmt.Errorf("no GGUF matching quantization %q found in Hugging Face repository", quant)
		}
		return nil, fmt.Errorf("no GGUF files found in Hugging Face repository")
	}
	if len(candidates) > 1 {
		return nil, &ambiguousGGUFError{candidates: candidates}
	}
	selected := groups[candidates[0]]
	files := make([]string, len(selected))
	for i, entry := range selected {
		files[i] = entry.Path
	}
	sort.Strings(files)
	if err := validateHFSplitFiles(files); err != nil {
		return nil, err
	}
	return files, nil
}

// validateHFSplitFiles rejects a partial or malformed split group before it
// can be loaded. A valid list must contain every shard exactly once, from 1
// through the advertised "of" count.
func validateHFSplitFiles(files []string) error {
	if len(files) == 0 {
		return fmt.Errorf("GGUF file set is empty")
	}
	first := hfSplitName.FindStringSubmatch(files[0])
	if first == nil {
		return nil
	}
	expected, err := strconv.Atoi(first[3])
	if err != nil || expected <= 0 || len(files) != expected {
		return fmt.Errorf("GGUF split is incomplete: found %d of %s shards", len(files), first[3])
	}
	seen := make(map[int]struct{}, len(files))
	for _, file := range files {
		match := hfSplitName.FindStringSubmatch(file)
		if match == nil || match[1] != first[1] || match[3] != first[3] {
			return fmt.Errorf("GGUF split has inconsistent shard names")
		}
		index, err := strconv.Atoi(match[2])
		if err != nil || index < 1 || index > expected {
			return fmt.Errorf("GGUF split has invalid shard name %q", file)
		}
		if _, duplicate := seen[index]; duplicate {
			return fmt.Errorf("GGUF split has duplicate shard %d", index)
		}
		seen[index] = struct{}{}
	}
	for index := 1; index <= expected; index++ {
		if _, ok := seen[index]; !ok {
			return fmt.Errorf("GGUF split is incomplete: shard %05d is missing", index)
		}
	}
	return nil
}

func ggufGroups(entries []hfTreeEntry) map[string][]hfTreeEntry {
	groups := map[string][]hfTreeEntry{}
	for _, entry := range entries {
		if entry.Type != "file" || !strings.HasSuffix(strings.ToLower(entry.Path), ".gguf") || !safeHFFilePath(entry.Path) {
			continue
		}
		key := splitGroupKey(entry.Path)
		groups[key] = append(groups[key], entry)
	}
	return groups
}

func ggufOptions(entries []hfTreeEntry, quantFilter string) []GGUFOption {
	groups := ggufGroups(entries)
	options := make([]GGUFOption, 0, len(groups))
	for file, files := range groups {
		quant := quantFromGGUFName(file)
		if quantFilter != "" && !strings.Contains(strings.ToLower(file), strings.ToLower(quantFilter)) {
			continue
		}
		option := GGUFOption{File: file, Quant: quant, Shards: len(files)}
		for _, entry := range files {
			option.SizeBytes += entry.Size
		}
		options = append(options, option)
	}
	sort.Slice(options, func(i, j int) bool { return options[i].File < options[j].File })
	return options
}

var hfQuantName = regexp.MustCompile(`(?i)(?:^|[-_.])((?:IQ|Q|TQ)\d(?:_[A-Z0-9]+)*|F16|BF16|F32)(?:[-_.]|$)`)

func quantFromGGUFName(name string) string {
	match := hfQuantName.FindStringSubmatch(name)
	if len(match) < 2 {
		return ""
	}
	return strings.ToUpper(match[1])
}

func formatHFSize(size int64) string {
	if size <= 0 {
		return "—"
	}
	const unit = int64(1024 * 1024)
	if size < 1024*unit {
		return fmt.Sprintf("%.1f MiB", float64(size)/float64(unit))
	}
	return fmt.Sprintf("%.2f GiB", float64(size)/float64(1024*unit))
}

var hfSplitName = regexp.MustCompile(`^(.*)-(\d{5})-of-(\d{5})\.gguf$`)

func splitGroupKey(path string) string {
	if m := hfSplitName.FindStringSubmatch(path); m != nil {
		return m[1] + ".gguf"
	}
	return path
}

func safeHFFilePath(path string) bool {
	for _, part := range strings.Split(path, "/") {
		if part == "" || part == "." || part == ".." {
			return false
		}
	}
	return true
}

func hfDownload(ctx context.Context, r hfReference, name string, cache hfCache, expectedSize int64, logw io.Writer, onProgress ProgressFunc, onBytes func(int64)) (string, error) {
	etag, err := hfFileETag(ctx, r, name)
	if err != nil {
		return "", err
	}
	blob := filepath.Join(cache.root, "blobs", etag)
	if st, err := os.Stat(blob); err == nil && st.Size() > 0 && (expectedSize <= 0 || st.Size() == expectedSize) {
		return blob, nil
	}
	incomplete := blob + ".incomplete"
	var offset int64
	if st, err := os.Stat(incomplete); err == nil {
		offset = st.Size()
		if expectedSize > 0 && offset > expectedSize {
			if err := os.Remove(incomplete); err != nil {
				return "", err
			}
			offset = 0
		}
	}
	if expectedSize > 0 && offset == expectedSize {
		if err := os.Rename(incomplete, blob); err != nil {
			return "", fmt.Errorf("finalize resumed download %s: %w", name, err)
		}
		return blob, nil
	}
	u := hfEndpoint() + "/" + hfRepositoryPath(r.Repository) + "/resolve/" + url.PathEscape(r.Revision) + "/" + strings.Join(escapeHFPath(name), "/") + "?download=true"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return "", err
	}
	hfAuth(req)
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("download %s: %w", name, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", hfStatusError(resp, name)
	}
	if offset > 0 && resp.StatusCode != http.StatusPartialContent {
		// Some mirrors ignore Range. Restart rather than appending a full
		// response and corrupting the model.
		offset = 0
	}
	if err := os.MkdirAll(filepath.Dir(blob), 0o755); err != nil {
		return "", err
	}
	if st, err := os.Stat(blob); err == nil && st.Size() > 0 {
		if err := os.Remove(blob); err != nil {
			return "", err
		}
	}
	flags := os.O_CREATE | os.O_WRONLY
	if offset > 0 {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	tmp, err := os.OpenFile(incomplete, flags, 0o644)
	if err != nil {
		return "", err
	}
	progress := &hfProgressWriter{out: logw, name: name, total: expectedSize, completed: offset, onBytes: onBytes, onEvent: onProgress}
	progress.Start()
	var bytesWritten int64
	if bytesWritten, err = io.Copy(tmp, io.TeeReader(resp.Body, progress)); err == nil {
		err = tmp.Close()
	} else {
		_ = tmp.Close()
	}
	if err != nil {
		progress.Finish(err)
		return "", fmt.Errorf("download %s: %w", name, err)
	}
	if expectedSize > 0 && offset+bytesWritten != expectedSize {
		err := fmt.Errorf("received %d of %d bytes", offset+bytesWritten, expectedSize)
		progress.Finish(err)
		return "", fmt.Errorf("download %s: %w", name, err)
	}
	if err := os.Rename(incomplete, blob); err != nil {
		progress.Finish(err)
		return "", fmt.Errorf("store %s: %w", name, err)
	}
	progress.Finish(nil)
	return blob, nil
}

// hfProgressWriter emits compact, periodically refreshed progress without
// requiring a terminal library; it also remains readable in redirected logs.
type hfProgressWriter struct {
	out       io.Writer
	name      string
	total     int64
	completed int64
	last      time.Time
	onBytes   func(int64)
	// onEvent mirrors the same throttled progress as onBytes/out, shaped for
	// callers that want structured (file, completed, total) updates instead
	// of parsing the human-readable log line.
	onEvent ProgressFunc
}

func (p *hfProgressWriter) Start() {
	p.last = time.Now()
	p.emitEvent(false, nil)
	if p.out == nil || p.out == io.Discard {
		return
	}
	if p.completed > 0 {
		fmt.Fprintf(p.out, "Hugging Face: resuming %s (%s already downloaded)\n", p.name, formatHFSize(p.completed))
	} else {
		fmt.Fprintf(p.out, "Hugging Face: downloading %s\n", p.name)
	}
}

func (p *hfProgressWriter) Write(data []byte) (int, error) {
	p.completed += int64(len(data))
	if p.onBytes != nil {
		p.onBytes(int64(len(data)))
	}
	if time.Since(p.last) >= 250*time.Millisecond {
		p.last = time.Now()
		p.emitEvent(false, nil)
		if p.out != nil && p.out != io.Discard {
			p.writeStatus()
		}
	}
	return len(data), nil
}

func (p *hfProgressWriter) Finish(err error) {
	p.emitEvent(true, err)
	if p.out == nil || p.out == io.Discard {
		return
	}
	p.writeStatus()
	if err == nil {
		fmt.Fprintln(p.out, " complete")
	} else {
		fmt.Fprintln(p.out, " interrupted; it will resume on the next run")
	}
}

func (p *hfProgressWriter) emitEvent(done bool, err error) {
	if p.onEvent == nil {
		return
	}
	p.onEvent(ProgressEvent{File: p.name, Completed: p.completed, Total: p.total, Done: done, Err: err})
}

func (p *hfProgressWriter) writeStatus() {
	if p.total > 0 {
		percent := min(100, 100*float64(p.completed)/float64(p.total))
		fmt.Fprintf(p.out, "\r  %.1f%% (%s / %s)", percent, formatHFSize(p.completed), formatHFSize(p.total))
		return
	}
	fmt.Fprintf(p.out, "\r  %s downloaded", formatHFSize(p.completed))
}

func hfFileETag(ctx context.Context, r hfReference, name string) (string, error) {
	u := hfEndpoint() + "/" + hfRepositoryPath(r.Repository) + "/resolve/" + url.PathEscape(r.Revision) + "/" + strings.Join(escapeHFPath(name), "/")
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, u, nil)
	if err != nil {
		return "", err
	}
	hfAuth(req)
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("read download metadata for %s: %w", name, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 400 {
		return "", hfStatusError(resp, name)
	}
	etag := safeHFETag(resp.Header.Get("X-Linked-ETag"))
	if etag == "" {
		etag = safeHFETag(resp.Header.Get("ETag"))
	}
	if etag == "" {
		return "", fmt.Errorf("download %s: Hugging Face did not provide a usable ETag", name)
	}
	return etag, nil
}

func safeHFETag(value string) string {
	value = strings.Trim(strings.TrimSpace(value), `"`)
	if value == "" || strings.ContainsAny(value, `/\\`) {
		return ""
	}
	for _, r := range value {
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '.' || r == '_' || r == '-') {
			return ""
		}
	}
	return value
}

// PlaceGGUFFiles links (or, when linking is unavailable, copies) each
// resolved GGUF shard into destDir under its original filename, in the order
// given. This lets a host such as a web UI make a Hub download show up in a
// local model directory without duplicating storage whenever destDir shares a
// volume with the Hugging Face cache, using the same link-then-copy fallback
// ResolveHuggingFaceModelContextWithOptions already relies on for its own
// blobs-to-snapshot layout. Callers choose destDir; PlaceGGUFFiles performs no
// sanitization of it, so it must already be a path the caller trusts (for
// example, derived from a fixed model directory plus a flattened, single
// path-component name rather than an unvalidated repository string).
func PlaceGGUFFiles(files []string, destDir string) ([]string, error) {
	if len(files) == 0 {
		return nil, fmt.Errorf("no GGUF files to place")
	}
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return nil, fmt.Errorf("create model directory: %w", err)
	}
	placed := make([]string, len(files))
	for i, file := range files {
		dest := filepath.Join(destDir, filepath.Base(file))
		if err := linkHFCacheFile(file, dest); err != nil {
			return nil, fmt.Errorf("place %s: %w", filepath.Base(file), err)
		}
		placed[i] = dest
	}
	return placed, nil
}

func linkHFCacheFile(blob, snapshotFile string) error {
	if err := os.MkdirAll(filepath.Dir(snapshotFile), 0o755); err != nil {
		return err
	}
	_ = os.Remove(snapshotFile)
	relative, err := filepath.Rel(filepath.Dir(snapshotFile), blob)
	if err == nil && os.Symlink(relative, snapshotFile) == nil {
		return nil
	}
	// Windows may lack symlink privileges. A hard link preserves the same
	// no-copy property; copy only when its filesystem does not support links.
	if err := os.Link(blob, snapshotFile); err == nil {
		return nil
	}
	in, err := os.Open(blob)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(snapshotFile)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(out, in)
	closeErr := out.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}

func escapeHFPath(path string) []string {
	parts := strings.Split(path, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return parts
}

func hfRepositoryPath(repository string) string {
	return strings.Join(escapeHFPath(repository), "/")
}
func hfAuth(req *http.Request) {
	if token := strings.TrimSpace(os.Getenv("HF_TOKEN")); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
}
func hfStatusError(resp *http.Response, target string) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	detail := strings.TrimSpace(string(body))
	if detail != "" {
		return fmt.Errorf("Hugging Face request for %s failed: %s: %s", target, resp.Status, detail)
	}
	return fmt.Errorf("Hugging Face request for %s failed: %s", target, resp.Status)
}

func cachedGGUFFiles(ctx context.Context, cache, quant string) ([]string, error) {
	entries, err := cachedHFGGUFEntries(ctx, cache)
	if err != nil {
		return nil, err
	}
	files, err := selectHFGGUF(entries, quant)
	if err != nil {
		return nil, err
	}
	for i := range files {
		files[i] = filepath.Join(cache, filepath.FromSlash(files[i]))
	}
	return files, nil
}

// cachedHFGGUFEntries reads snapshot links using os.Stat, which follows
// symlinks into blobs and deliberately ignores stale links from a manually
// damaged cache. The resulting sizes make offline --hf-list as informative as
// an online repository listing.
func cachedHFGGUFEntries(ctx context.Context, cache string) ([]hfTreeEntry, error) {
	entries := []hfTreeEntry{}
	err := filepath.WalkDir(cache, func(path string, d os.DirEntry, err error) error {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(strings.ToLower(path), ".gguf") {
			info, statErr := os.Stat(path)
			if statErr != nil || !info.Mode().IsRegular() {
				return nil
			}
			rel, _ := filepath.Rel(cache, path)
			entries = append(entries, hfTreeEntry{Path: filepath.ToSlash(rel), Type: "file", Size: info.Size()})
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return entries, nil
}

func completeCachedGGUFOptions(entries []hfTreeEntry, options []GGUFOption) []GGUFOption {
	groups := ggufGroups(entries)
	complete := make([]GGUFOption, 0, len(options))
	for _, option := range options {
		files := groups[option.File]
		paths := make([]string, len(files))
		for i, entry := range files {
			paths[i] = entry.Path
		}
		sort.Strings(paths)
		if validateHFSplitFiles(paths) == nil {
			complete = append(complete, option)
		}
	}
	return complete
}
