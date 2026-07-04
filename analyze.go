package gopherllm

// GGUF analysis: a structural report over a parsed file (no weights needed),
// vocabulary search, and embedding-space token neighborhoods (weights
// needed). Exposed on the CLI as --analyze, --find-token, and
// --token-neighbors.

import (
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
)

// DTypeStat summarizes one tensor dtype's share of a model.
type DTypeStat struct {
	Type    GGMLType
	Tensors int
	Bytes   int64
	Params  int64
}

// TensorStat identifies one tensor and its size for the largest-tensors list.
type TensorStat struct {
	Name   string
	Type   GGMLType
	Params int64
	Bytes  int64
}

// Analysis is a structural report over a GGUF header: identity, geometry,
// quantization mix, tokenizer properties, and derived estimates. Everything
// here comes from the header and metadata — no tensor data is read — so
// analyzing a multi-gigabyte file is instant.
type Analysis struct {
	Architecture string
	Name         string
	Version      uint32
	Supported    bool

	Params        int64 // total weight count across all tensors
	FileBytes     int64 // sum of tensor bytes (excludes the small header)
	BitsPerWeight float64

	Layers        int
	Dim           int
	HiddenDim     int
	Heads         int
	KVHeads       int
	HeadDim       int
	VocabSize     int
	ContextLength int
	SlidingWindow int
	SWALayers     int // layers using the sliding window (0 = none, -1 = all)

	RopeTheta       float32
	RopeScalingType string

	TokenizerModel string
	TokenizerPre   string
	ChatTemplate   bool
	TemplateKind   string
	BOSID, EOSID   uint32

	DTypes         []DTypeStat  // sorted by byte share, descending
	LargestTensors []TensorStat // top 5 by bytes

	// KVCacheBytesAtFullContext / At4K estimate the f32 KV cache footprint.
	KVCacheBytesAtFullContext int64
	KVCacheBytesAt4K          int64
}

// AnalyzeGGUF builds an Analysis from a parsed GGUF. tok may be nil (tokenizer
// metadata is then reported from raw metadata only, without template
// detection).
func AnalyzeGGUF(g *GGUFFile, tok *Tokenizer) *Analysis {
	cfg := ConfigFromGGUF(g)
	arch, _ := g.GetString("general.architecture")
	name, _ := g.GetString("general.name")

	a := &Analysis{
		Architecture:    arch,
		Name:            name,
		Version:         g.Version,
		Supported:       ArchitectureSupported(arch),
		Layers:          cfg.NLayers,
		Dim:             cfg.Dim,
		HiddenDim:       cfg.HiddenDim,
		Heads:           cfg.NHeads,
		KVHeads:         cfg.NKVHeads,
		HeadDim:         cfg.HeadDim,
		VocabSize:       cfg.VocabSize,
		ContextLength:   cfg.MaxSeqLen,
		SlidingWindow:   cfg.SlidingWindow,
		RopeTheta:       cfg.RopeTheta,
		RopeScalingType: cfg.RopeScalingType,
	}
	if cfg.SlidingWindow > 0 {
		if cfg.SWAPattern == nil {
			a.SWALayers = -1
		} else {
			for _, swa := range cfg.SWAPattern {
				if swa {
					a.SWALayers++
				}
			}
		}
	}

	byType := map[GGMLType]*DTypeStat{}
	for _, t := range g.Tensors {
		numel := int64(t.Numel())
		bytes64, ok := t.DType.DataSize(t.Numel())
		var bytes int64
		if ok {
			bytes = int64(bytes64)
		}
		st := byType[t.DType]
		if st == nil {
			st = &DTypeStat{Type: t.DType}
			byType[t.DType] = st
		}
		st.Tensors++
		st.Params += numel
		st.Bytes += bytes
		a.Params += numel
		a.FileBytes += bytes
		a.LargestTensors = append(a.LargestTensors, TensorStat{Name: t.Name, Type: t.DType, Params: numel, Bytes: bytes})
	}
	for _, st := range byType {
		a.DTypes = append(a.DTypes, *st)
	}
	sort.Slice(a.DTypes, func(i, j int) bool { return a.DTypes[i].Bytes > a.DTypes[j].Bytes })
	sort.Slice(a.LargestTensors, func(i, j int) bool { return a.LargestTensors[i].Bytes > a.LargestTensors[j].Bytes })
	if len(a.LargestTensors) > 5 {
		a.LargestTensors = a.LargestTensors[:5]
	}
	if a.Params > 0 {
		a.BitsPerWeight = float64(a.FileBytes) * 8 / float64(a.Params)
	}

	// KV cache: per position, every layer stores NKVHeads*HeadDim keys plus
	// NKVHeads*ValueDim values as f32.
	perPos := int64(cfg.NLayers) * int64(cfg.NKVHeads) * int64(cfg.HeadDim+cfg.ValueDim) * 4
	a.KVCacheBytesAtFullContext = perPos * int64(cfg.MaxSeqLen)
	a.KVCacheBytesAt4K = perPos * int64(min(4096, cfg.MaxSeqLen))

	if v, ok := g.Metadata["tokenizer.ggml.model"]; ok {
		a.TokenizerModel, _ = v.AsString()
	}
	if v, ok := g.Metadata["tokenizer.ggml.pre"]; ok {
		a.TokenizerPre, _ = v.AsString()
	}
	_, a.ChatTemplate = g.Metadata["tokenizer.chat_template"]
	if tok != nil {
		a.BOSID, a.EOSID = tok.BOSID, tok.EOSID
		a.TemplateKind = (&Runner{gguf: g, tok: tok}).chatTemplateKind()
	}
	return a
}

// WriteText renders the analysis as a human-readable report.
func (a *Analysis) WriteText(w io.Writer) {
	gb := func(b int64) string { return fmt.Sprintf("%.2f GB", float64(b)/(1024*1024*1024)) }
	fmt.Fprintf(w, "architecture:   %s (supported: %v)\n", a.Architecture, a.Supported)
	if a.Name != "" {
		fmt.Fprintf(w, "name:           %s\n", a.Name)
	}
	fmt.Fprintf(w, "gguf version:   %d\n", a.Version)
	fmt.Fprintf(w, "parameters:     %.2fB (%s tensor data, %.2f bits/weight)\n", float64(a.Params)/1e9, gb(a.FileBytes), a.BitsPerWeight)
	fmt.Fprintf(w, "geometry:       %d layers, dim %d, hidden %d, heads %d/%d (head_dim %d)\n", a.Layers, a.Dim, a.HiddenDim, a.Heads, a.KVHeads, a.HeadDim)
	fmt.Fprintf(w, "vocab/context:  %d tokens, %d context\n", a.VocabSize, a.ContextLength)
	if a.SlidingWindow > 0 {
		layers := "all"
		if a.SWALayers >= 0 {
			layers = fmt.Sprintf("%d/%d", a.SWALayers, a.Layers)
		}
		fmt.Fprintf(w, "sliding window: %d tokens on %s layers\n", a.SlidingWindow, layers)
	}
	fmt.Fprintf(w, "rope:           theta %.0f", a.RopeTheta)
	if a.RopeScalingType != "" {
		fmt.Fprintf(w, ", scaling %s", a.RopeScalingType)
	}
	fmt.Fprintln(w)
	fmt.Fprintf(w, "tokenizer:      %s", a.TokenizerModel)
	if a.TokenizerPre != "" {
		fmt.Fprintf(w, " (pre: %s)", a.TokenizerPre)
	}
	fmt.Fprintf(w, ", BOS %d, EOS %d\n", a.BOSID, a.EOSID)
	kind := a.TemplateKind
	if kind == "" {
		kind = "plain fallback"
	}
	fmt.Fprintf(w, "chat template:  embedded=%v, detected=%s\n", a.ChatTemplate, kind)
	fmt.Fprintf(w, "kv cache (f32): %s at full context, %s at 4K\n", gb(a.KVCacheBytesAtFullContext), gb(a.KVCacheBytesAt4K))
	fmt.Fprintln(w, "tensor types:")
	for _, st := range a.DTypes {
		share := float64(0)
		if a.FileBytes > 0 {
			share = 100 * float64(st.Bytes) / float64(a.FileBytes)
		}
		fmt.Fprintf(w, "  %-6s %4d tensors  %8s  %5.1f%%\n", st.Type, st.Tensors, gb(st.Bytes), share)
	}
	fmt.Fprintln(w, "largest tensors:")
	for _, t := range a.LargestTensors {
		fmt.Fprintf(w, "  %-40s %-6s %8s\n", t.Name, t.Type, gb(t.Bytes))
	}
}

// TokenMatch is one vocabulary entry, with Score meaning depending on the
// producing call: SearchTokens leaves it 0; NearestTokens sets cosine
// similarity.
type TokenMatch struct {
	ID    uint32
	Text  string
	Score float32
}

// SearchTokens finds vocabulary entries whose decoded text contains query
// (case-insensitive), exact matches first, capped at limit (<=0 means 100).
// Both the raw vocab form (with byte/▁ markers) and the human-decoded form
// are searched, so "hello", " hello", and "▁hello"-style pieces all match.
func SearchTokens(tok *Tokenizer, query string, limit int) []TokenMatch {
	if limit <= 0 {
		limit = 100
	}
	needle := strings.ToLower(query)
	var exact, partial []TokenMatch
	for id := range tok.Vocab {
		decoded := tok.DecodeToken(uint32(id))
		raw := tok.Vocab[id]
		dl, rl := strings.ToLower(decoded), strings.ToLower(raw)
		switch {
		case dl == needle || rl == needle || strings.TrimSpace(dl) == needle:
			exact = append(exact, TokenMatch{ID: uint32(id), Text: decoded})
		case strings.Contains(dl, needle) || strings.Contains(rl, needle):
			partial = append(partial, TokenMatch{ID: uint32(id), Text: decoded})
		}
	}
	out := append(exact, partial...)
	if len(out) > limit {
		out = out[:limit]
	}
	return out
}

// tokenEmbedding returns the model's token-embedding weight.
func (r *Runner) tokenEmbedding() Weight {
	switch r.kind {
	case loadedGptOss:
		return r.gptOss.Standard.TokenEmbd
	case loadedGemma4:
		return r.gemma4.TokenEmbd
	default:
		return r.standard.TokenEmbd
	}
}

// NearestTokens returns the k vocabulary tokens whose embedding vectors are
// most cosine-similar to token id's embedding (excluding id itself) — the
// "which tokens does the model treat as related?" view. It dequantizes and
// scans the full embedding table (parallelized), so expect O(vocab*dim) work:
// fractions of a second for 32K vocabularies, a few seconds at 262K.
func (r *Runner) NearestTokens(id uint32, k int) ([]TokenMatch, error) {
	if k <= 0 {
		k = 10
	}
	dim := r.config.Dim
	vocab := r.config.VocabSize
	if int(id) >= vocab {
		return nil, fmt.Errorf("token id %d out of range (vocab %d)", id, vocab)
	}
	embd := r.tokenEmbedding()
	ref := make([]float32, dim)
	refBuf := ref
	embd.RowInto(int(id), dim, &refBuf)
	refNorm := DotF32(ref, ref)
	if refNorm <= 0 {
		return nil, fmt.Errorf("token %d has a zero embedding", id)
	}

	scores := make([]float32, vocab)
	parallelChunks(vocab, func(start, end int) {
		row := make([]float32, dim)
		for t := start; t < end; t++ {
			buf := row
			embd.RowInto(t, dim, &buf)
			denom := DotF32(buf, buf) * refNorm
			if denom <= 0 {
				scores[t] = -1
				continue
			}
			scores[t] = DotF32(ref, buf) / float32(math.Sqrt(float64(denom)))
		}
	})

	type cand struct {
		id    int
		score float32
	}
	cands := make([]cand, 0, vocab-1)
	for t := 0; t < vocab; t++ {
		if uint32(t) == id {
			continue
		}
		cands = append(cands, cand{t, scores[t]})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].score > cands[j].score })
	if len(cands) > k {
		cands = cands[:k]
	}
	out := make([]TokenMatch, len(cands))
	for i, c := range cands {
		out[i] = TokenMatch{ID: uint32(c.id), Text: r.tok.DecodeToken(uint32(c.id)), Score: c.score}
	}
	return out, nil
}

// NearestTokens on the Model API. token may be a numeric id or a literal
// token text (resolved via exact vocabulary search).
func (m *Model) NearestTokens(token string, k int) ([]TokenMatch, error) {
	id, err := m.resolveTokenID(token)
	if err != nil {
		return nil, err
	}
	return m.r.NearestTokens(id, k)
}

func (m *Model) resolveTokenID(token string) (uint32, error) {
	var id uint32
	if _, err := fmt.Sscanf(token, "%d", &id); err == nil && fmt.Sprint(id) == strings.TrimSpace(token) {
		return id, nil
	}
	if id, ok := m.r.tok.SpecialID(token); ok {
		return id, nil
	}
	matches := SearchTokens(m.r.tok, token, 1)
	if len(matches) == 0 {
		return 0, fmt.Errorf("no token matches %q", token)
	}
	return matches[0].ID, nil
}