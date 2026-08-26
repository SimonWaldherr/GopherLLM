package gopherllm

import (
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
)

// ErrRunnerClosed is returned when an operation begins after Runner.Close.
// A Runner may be shared between goroutines; the generation lock makes a
// concurrent Close deterministic rather than letting a waiter reach cleared
// or unmapped weights after Close wins the lock.
var ErrRunnerClosed = errors.New("runner closed")

type LoadInfo struct {
	FileSizeBytes int
	LoadTime      time.Duration
	// OutOfCore reports whether this model was loaded through the CPU-only
	// demand-paged path. It does not promise a fixed RSS cap: the OS controls
	// mmap page residency and may retain or evict file-backed pages as needed.
	OutOfCore bool
}

type LoadOptions struct {
	PrepareQuantized bool
	UseMetal         bool
	// BorrowQuantized keeps packed quantized tensors as views into the input
	// byte slice instead of copying each tensor. The Runner retains the
	// backing storage for as long as those weights are live, so callers must
	// not mutate the input while the Runner is in use. This is useful for
	// browser/WASM loads, where the input has already been copied into the Go
	// heap and a second full model-sized copy can exceed the address-space
	// budget. It is disabled for byte-backed loads with UseMetal because
	// Metal's no-copy buffers may retain only OS-mapped memory, never a Go
	// heap pointer.
	BorrowQuantized bool
	// OutOfCore keeps scalar and quantized matrices as views of a real mmap,
	// disables GPU/prepared copies, and avoids prewarming sparse expert banks.
	// It requires a single-file GGUF opened from a filesystem path.
	OutOfCore bool
	// Prefault selects the mmap warm-up policy. The zero value preserves the
	// historical full warm-up; OutOfCore changes its effective policy to core.
	Prefault  MmapPrefaultMode
	LogWriter io.Writer
	// VisionProjectorPath/VisionProjectorBytes optionally load a companion
	// Pixtral-style "mmproj" vision-encoder GGUF alongside the text decoder
	// (mutually exclusive; Bytes takes precedence if both are set). This
	// attaches independently of the architecture dispatch below — it is
	// never a loadedKind of its own, since a vision encoder is a companion
	// file, not a distinct text-decoder graph shape. See Runner.vision.
	VisionProjectorPath  string
	VisionProjectorBytes []byte
}

type loadedKind int

const (
	loadedStandard loadedKind = iota
	loadedGptOss
	loadedGemma4
	loadedNemotronH
	loadedMamba2
	loadedBERT
	loadedQwen35
)

// Runner is a fully loaded model ready to generate: parsed GGUF header,
// tokenizer, config, and weights (one of the three kind-specific sets).
// Generations and embeddings are serialized by genLock — a Runner is safe to
// share across goroutines (the HTTP server does), but runs one request at a
// time. modelMu keeps borrowed weight storage alive for every active reader;
// Close takes it exclusively, then resource-consuming calls begun afterwards
// return ErrRunnerClosed.
type Runner struct {
	gguf      *GGUFFile
	arch      string
	tok       *Tokenizer
	config    Config
	kind      loadedKind
	standard  ModelWeights
	gptOss    GptOssWeights
	gemma4    Gemma4Weights
	nemotronH NemotronHWeights
	mamba2    Mamba2Weights
	bert      BERTWeights
	qwen35    Qwen35Weights
	// modelMu protects the lifecycle of all CPU/GPU weight storage. The
	// lock order for resource-consuming work is modelMu, then genLock, then
	// visionMu (and finally visionCacheMu when needed).
	modelMu sync.RWMutex
	genLock sync.Mutex
	// closed is protected by modelMu. Resource-consuming public operations
	// acquire a shared model lease before reading weights, so a call queued
	// behind Close returns ErrRunnerClosed rather than touching containers that
	// Close cleared.
	closed         bool
	workspaceCache *KVCache
	workspaceBuf   *DecodeBuffer
	bertScratch    bertEmbeddingScratch
	prefixCache    prefixCacheState
	// mistralPrefixCacheMu protects the opt-in render-time cache used by the
	// Mistral/Ministral chat renderer. Unlike prefixCache, it is intentionally
	// reachable from concurrent PrepareChatContext calls, which do not take
	// genLock. It holds tokenized static prompt text only, never model state.
	mistralPrefixCacheMu     sync.Mutex
	mistralPrefixRenderCache mistralPromptPrefixCache
	mappedFile               *MmapFile
	// extraMappedFiles holds the additional shard mappings of an out-of-core
	// split GGUF. Those models have no single mappedFile: every weight is a
	// view into one of these, so all of them must outlive the Runner and all
	// of them are closed by Close.
	extraMappedFiles []*MmapFile
	outOfCore        bool
	// vision, when non-nil, is a loaded Pixtral-style vision encoder paired
	// with this Runner's text decoder (see LoadOptions.VisionProjector*).
	// Independent of kind/loadedKind — only chat-template renderers that
	// know how to splice image placeholders (currently Mistral-family)
	// consult it.
	vision           *PixtralVisionWeights
	visionConfig     PixtralVisionConfig
	visionMappedFile *MmapFile
	// visionMu protects the vision tower and its dedicated mapping. Unlike
	// generation, PrepareChatContext is intentionally usable concurrently and
	// can encode an image through renderMessages, so genLock alone cannot
	// protect a Close from releasing the tower underneath that reader.
	//
	// Lock order is modelMu, genLock, then visionMu when more than one is
	// needed. Never call HasVision while already holding a visionMu read lock:
	// a waiting writer makes recursive RLock deadlock.
	visionMu sync.RWMutex
	// visionCacheMu guards the three vision-cache fields below.
	//
	// Deliberately its own lock rather than genLock: PrepareChatContext is
	// exported, takes no lock, and reaches this cache through
	// renderMessages -> renderMistralInstMessages -> encodeChatImage. An
	// application that plans a context window while a generation is running
	// -- which is an ordinary thing to do, and what the server does -- would
	// otherwise hit concurrent map writes, which Go turns into a hard crash
	// rather than a mere data race.
	//
	// Never take genLock while holding this; nothing here needs to.
	visionCacheMu sync.Mutex
	// visionImageCache memoizes EncodeImagePixtral results, keyed by an fnv
	// hash of the image bytes.
	//
	// It deliberately survives across generation calls. Within one call it
	// covers the same-image-rendered-twice case; across calls it covers the
	// far more expensive one, which is how people actually use vision: every
	// follow-up question about an already-attached picture re-sends that
	// picture in the history, and re-running a 24-block tower over it costs
	// tens of seconds. Encoding it once per conversation instead of once per
	// turn is the single largest saving available on a multi-turn image chat.
	//
	// Persisting is only safe because it is bounded (see visionCachePut): a
	// live camera feeds a unique image per frame, and an unbounded map of
	// multi-megabyte float tensors would turn that into steadily rising GC
	// pressure. Bounding evicts those frames as fast as they arrive while
	// still holding the handful of stills a conversation refers back to.
	//
	// Entries are read-only once stored: both consumers copy out of them
	// (copy(X[t][:dim], emb) in forward_batch.go and model.go), never in
	// place, so handing the same slices to several calls is sound.
	visionImageCache map[uint64]visionImageCacheEntry
	// visionImageOrder is the eviction order, least recently used first.
	// A slice scan is the right structure at these sizes: it holds at most
	// visionImageCacheMaxEntries keys, so a real LRU's bookkeeping would cost
	// more than the linear walk it replaces.
	visionImageOrder []uint64
	// visionImageFloats is the resident float32 count across all entries,
	// tracked incrementally so eviction never has to re-walk the map.
	visionImageFloats int
}

const (
	// Caps on the vision cache. Entries scale with the image's token count
	// (mergedTokens x ProjectionDim), which for a 1024-token image at
	// ProjectionDim 3072 is about 12 MB, so a count-only bound could still
	// retain hundreds of megabytes. Bound both.
	visionImageCacheMaxEntries = 8
	visionImageCacheMaxFloats  = 16 << 20 // 64 MiB of float32
)

type visionImageCacheEntry struct {
	embeds                 [][]float32
	mergedRows, mergedCols int
}

// HasVision reports whether this Runner has a paired vision encoder loaded.
func (r *Runner) HasVision() bool {
	if r == nil {
		return false
	}
	r.visionMu.RLock()
	defer r.visionMu.RUnlock()
	return r.vision != nil
}

// ArchitectureSupported reports whether the loader accepts this
// general.architecture value. Notes on specific families:
//
//   - qwen3 (incl. the DeepSeek-R1-0528 Qwen3 distills): the qwen2 graph plus
//     per-head QK-norm, which loads via the optional attn_q_norm/attn_k_norm
//     tensors and applies exactly as for Gemma 3/4.
//   - Mixtral/Llama-MoE, qwen2moe, qwen3moe, gpt-oss, and DeepSeek/Kimi use
//     sparse experts when their router tensors are present. deepseek2 uses a
//     dedicated MLA attention path and its sigmoid/noaux shared-expert router.
//   - qwen35/qwen35moe use the experimental native Gated-DeltaNet hybrid
//     loader. It has focused graph and local-GGUF smoke coverage; a trailing
//     one-layer Qwen MTP draft head can be enabled for exact greedy
//     speculation. Cross-runtime logit parity and vision remain out of scope;
//     qwen35moe shares the sparse-expert trunk tensor path.
//   - Phi-3, dense Granite, EXAONE, and InternLM2 use the same pre-norm, RoPE, GQA and
//     SwiGLU graph as the standard loader. Their architecture-specific scales
//     are read from GGUF metadata by ConfigFromGGUF.
//   - SmolLM3 uses the standard dense graph with its every-fourth-layer RoPE
//     omission. EXAONE 4 uses per-head QK norm, post-attention/post-FFN norm,
//     and its local/global attention schedule.
//   - OLMo 2 and OLMo 3 both declare olmo2. They use full-projection QK norm,
//     post-norm residual branches, and (for OLMo 3) separate local/global RoPE.
//   - Phi-2 uses its native biased LayerNorm, parallel attention/exact-GELU
//     MLP branches, mandatory projection biases, and biased vocabulary head.
//   - Devstral and Mistral-Small GGUFs usually declare llama or mistral3;
//     their [INST]/Tekken behavior is picked up from tokenizer metadata, not
//     the arch string.
//
// This is an exact-label check. The loader does not call it on the raw
// general.architecture string but on the label returned by
// ResolveArchitecture, which first normalizes spelling variants (Hugging Face
// class names, hyphen/underscore forms) and falls back to the hyperparameter
// namespace actually present in the file, so a mislabeled GGUF whose contents
// are a supported architecture still loads.
func ArchitectureSupported(arch string) bool {
	switch arch {
	case "llama", "llama2", "llama3", "mistral", "mistral3", "ministral", "mixtral",
		"qwen2", "qwen2moe", "qwen3", "qwen3moe", "qwen35", "qwen35moe", "deepseek2", "kimi_k2", "phi3", "granite", "granitemoe", "exaone", "internlm2", "stablelm", "gpt-oss", "gemma", "gemma2", "gemma3", "gemma4", "nemotron_h", "nemotron_h_moe", "mamba2", "bert", "nomic-bert":
		return true
	case "smollm3", "exaone4":
		return true
	case "gpt2", "gptneox", "gptj", "bloom", "mpt", "falcon", "starcoder", "starcoder2":
		return true
	case "chatglm", "glm4", "command-r", "minicpm":
		return true
	case "olmo2":
		return true
	case "phi2":
		return true
	default:
		return false
	}
}

// RunnerFromGGUFBytes loads a model from an in-memory GGUF, copying quantized
// tensors into owned memory. It is silent; use Open with WithLogWriter for
// load-progress diagnostics.
func RunnerFromGGUFBytes(data []byte) (*Runner, error) {
	return RunnerFromGGUFBytesWithOptions(data, LoadOptions{})
}

func RunnerFromGGUFBytesWithOptions(data []byte, options LoadOptions) (*Runner, error) {
	if options.OutOfCore {
		return nil, fmt.Errorf("out-of-core loading requires RunnerFromPathWithOptions: byte-backed models already reside in memory")
	}
	// Metal's zero-copy path retains the supplied address in an Objective-C
	// buffer. A byte-backed GGUF is Go-managed memory, not an OS mapping, so
	// it must be copied before crossing that ownership boundary.
	if options.UseMetal {
		options.BorrowQuantized = false
	}
	return runnerFromGGUFBytes(data, options.BorrowQuantized, options)
}

// RunnerFromGGUFBytesWithVision loads a text-decoder GGUF plus a paired
// Pixtral-style vision-encoder "mmproj" GGUF, both from bytes. Sugar over
// LoadOptions.VisionProjectorBytes.
func RunnerFromGGUFBytesWithVision(textData, visionData []byte, options LoadOptions) (*Runner, error) {
	options.VisionProjectorBytes = visionData
	return RunnerFromGGUFBytesWithOptions(textData, options)
}

func runnerFromGGUFBytes(data []byte, borrowQuantized bool, options LoadOptions) (*Runner, error) {
	if err := validateLoadOptions(options); err != nil {
		return nil, err
	}
	if options.OutOfCore && !borrowQuantized {
		return nil, fmt.Errorf("out-of-core loading requires a memory-mapped model file")
	}
	gguf, err := ParseGGUF(data)
	if err != nil {
		return nil, err
	}
	return runnerFromParsedGGUF(data, gguf, borrowQuantized, options)
}

// runnerFromParsedGGUF builds a Runner from an already-parsed GGUFFile plus
// the byte slice its Tensors' Offsets are relative to (via DataOffset). It is
// split out from runnerFromGGUFBytes so the split-file loader (gguf_split.go)
// can hand in a synthetic GGUFFile assembled from multiple shard files
// without needing a real single-file byte stream to re-parse.
func runnerFromParsedGGUF(data []byte, gguf *GGUFFile, borrowQuantized bool, options LoadOptions) (*Runner, error) {
	if err := validateLoadOptions(options); err != nil {
		return nil, err
	}
	logw := options.LogWriter
	if logw == nil {
		logw = io.Discard
	}
	fmt.Fprintf(logw, "GGUF v%d - %d tensors, %d metadata entries\n", gguf.Version, len(gguf.Tensors), len(gguf.Metadata))
	declaredArch, _ := gguf.GetString("general.architecture")
	declaredArch = strings.TrimSpace(declaredArch)
	arch, hparamNS := ResolveArchitecture(gguf)
	switch {
	case declaredArch == "" && namespaceHasHParams(gguf, hparamNS):
		fmt.Fprintf(logw, "No general.architecture in metadata; detected %q from the %s.* hyperparameter namespace\n", arch, hparamNS)
	case declaredArch == "":
		fmt.Fprintf(logw, "No general.architecture in metadata and no hyperparameter namespace found; assuming %q\n", arch)
	case arch != declaredArch:
		fmt.Fprintf(logw, "Architecture %q is not a label GopherLLM knows; loading as %q (hyperparameters under %s.*)\n", declaredArch, arch, hparamNS)
	case hparamNS != arch:
		fmt.Fprintf(logw, "Architecture %s: reading hyperparameters from the %s.* metadata namespace\n", arch, hparamNS)
	}
	if !ArchitectureSupported(arch) {
		if isVisionProjectorGGUF(gguf, arch) {
			return nil, fmt.Errorf("this GGUF is a CLIP/mmproj vision projector, not a text model; load it alongside a text-model GGUF via LoadOptions.VisionProjectorPath (CLI: --mmproj)")
		}
		if hparamNS != arch {
			return nil, fmt.Errorf("unsupported architecture: %s (hyperparameters found under %s.*)", arch, hparamNS)
		}
		return nil, fmt.Errorf("unsupported architecture: %s", arch)
	}
	tok, err := TokenizerFromMetadata(gguf.Metadata)
	if err != nil {
		return nil, err
	}
	r := &Runner{gguf: gguf, arch: arch, tok: tok, outOfCore: options.OutOfCore}
	if options.OutOfCore {
		fmt.Fprintln(logw, "Out-of-core: CPU mmap mode; sparse experts remain demand-paged (Metal and prepared quantization disabled)")
	}
	switch arch {
	case "bert", "nomic-bert":
		config, weights, err := LoadBERTModel(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw, options.OutOfCore)
		if err != nil {
			return nil, err
		}
		r.config, r.bert, r.kind = config, weights, loadedBERT
	case "nemotron_h", "nemotron_h_moe":
		config, weights, err := LoadNemotronHModel(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw, options.OutOfCore)
		if err != nil {
			return nil, err
		}
		r.config, r.nemotronH, r.kind = config, weights, loadedNemotronH
	case "mamba2":
		config, weights, err := LoadMamba2Model(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw, options.OutOfCore)
		if err != nil {
			return nil, err
		}
		r.config, r.mamba2, r.kind = config, weights, loadedMamba2
	case "qwen35", "qwen35moe":
		// Text-only decode and the optional exact-greedy MTP draft head are
		// implemented and covered by focused graph tests. Keep the remaining
		// unsupported capabilities explicit instead of claiming general Qwen
		// feature parity.
		fmt.Fprintln(logw, "Warning: qwen35 support is experimental: text-only decode and greedy MTP speculation are implemented; vision and cross-runtime logit-parity validation are pending (see qwen35.go)")
		config, weights, err := LoadQwen35Model(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw, options.OutOfCore)
		if err != nil {
			return nil, err
		}
		r.config, r.qwen35, r.kind = config, weights, loadedQwen35
	case "gpt-oss":
		config, weights, err := LoadGptOssModel(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw, options.OutOfCore)
		if err != nil {
			return nil, err
		}
		r.config, r.gptOss, r.kind = config, weights, loadedGptOss
	case "gemma", "gemma2", "gemma3", "gemma4":
		config, weights, err := LoadGemma4Model(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw, options.OutOfCore)
		if err != nil {
			return nil, err
		}
		if arch == "gemma4" && weights.Native {
			features := []string{"proportional RoPE", "K-as-V", "per-layer output scales", "native turn template"}
			if len(weights.MoE) > 0 {
				features = append(features, "shared-dense/MoE FFN")
			}
			if weights.PerLayer != nil {
				features = append(features, "per-layer embeddings")
			}
			if nativeGemma4KVCacheLayerCount(weights) < len(weights.Layers) {
				features = append(features, "shared KV cache")
			}
			fmt.Fprintf(logw, "Gemma 4: native SWA/global decoder enabled (%s)\n", strings.Join(features, ", "))
		} else {
			fmt.Fprintf(logw, "Warning: %s support is experimental; dense decoder support has not had cross-runtime logit-parity validation\n", arch)
		}
		r.config, r.gemma4, r.kind = config, weights, loadedGemma4
	default:
		config, weights, err := LoadModel(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw, options.OutOfCore)
		if err != nil {
			return nil, err
		}
		r.config, r.standard, r.kind = config, weights, loadedStandard
	}

	if len(options.VisionProjectorBytes) > 0 || options.VisionProjectorPath != "" {
		visionData := options.VisionProjectorBytes
		var visionFile *MmapFile
		if len(visionData) == 0 {
			mmap, err := OpenMmap(options.VisionProjectorPath)
			if err != nil {
				return nil, fmt.Errorf("loading vision projector: %w", err)
			}
			visionData, visionFile = mmap.Bytes(), mmap
		}
		visionGGUF, err := ParseGGUFQuiet(visionData)
		if err != nil {
			if visionFile != nil {
				_ = visionFile.Close()
			}
			return nil, fmt.Errorf("loading vision projector: %w", err)
		}
		var vc PixtralVisionConfig
		var vw PixtralVisionWeights
		// A Metal zero-copy buffer may retain only an OS-mapped address. The
		// text model's effective borrow mode does not establish that the
		// companion uses a mapping: a second OpenMmap can fall back to a Go
		// heap read, and VisionProjectorBytes is always heap-backed.
		visionBorrow := borrowQuantized && (!options.UseMetal || (visionFile != nil && visionFile.IsMapped()))
		if visionBorrow {
			vc, vw, err = loadPixtralVisionModel(visionData, visionGGUF, options.UseMetal, true, logw)
		} else {
			vc, vw, err = LoadPixtralVisionModel(visionData, visionGGUF, options.UseMetal, logw)
		}
		if err != nil {
			if visionFile != nil {
				_ = visionFile.Close()
			}
			return nil, fmt.Errorf("loading vision projector: %w", err)
		}
		if len(vw.ImgBreak) != r.config.Dim {
			releasePixtralVisionWeights(&vw)
			if visionFile != nil {
				_ = visionFile.Close()
			}
			return nil, fmt.Errorf("loading vision projector: img_break width %d does not match text model dim %d", len(vw.ImgBreak), r.config.Dim)
		}
		r.vision, r.visionConfig, r.visionMappedFile = &vw, vc, visionFile
		fmt.Fprintf(logw, "Vision: loaded Pixtral-style encoder (%d layers, %d-dim, %dx merge)\n", vc.BlockCount, vc.EmbeddingLength, vc.SpatialMergeSize)
	}

	return r, nil
}

// RunnerFromPath memory-maps a GGUF file and loads it with zero-copy borrowed
// quantized weights. Silent; prefer Open, which adds context support,
// configurable logging, and a higher-level Model wrapper.
func RunnerFromPath(path string) (*Runner, LoadInfo, error) {
	return RunnerFromPathWithOptions(path, LoadOptions{})
}

func RunnerFromPathWithOptions(path string, options LoadOptions) (*Runner, LoadInfo, error) {
	if err := validateLoadOptions(options); err != nil {
		return nil, LoadInfo{}, err
	}
	t0 := time.Now()
	mmap, err := OpenMmap(path)
	if err != nil {
		return nil, LoadInfo{}, fmt.Errorf("failed to open model: %w", err)
	}
	if options.OutOfCore && !mmap.IsMapped() {
		_ = mmap.Close()
		return nil, LoadInfo{}, fmt.Errorf("out-of-core loading requires an OS memory map; this file fell back to an in-memory read")
	}
	if header, herr := ParseGGUFQuiet(mmap.Bytes()); herr == nil {
		if _, count, ok := splitInfo(header); ok && count > 1 {
			r, modelBytes, err := loadSplitRunner(path, header, mmap, options)
			if err != nil {
				return nil, LoadInfo{}, err
			}
			// Out-of-core keeps every shard mapped and reports the mapped
			// size; the merging path reports the size of the copy it made.
			return r, LoadInfo{FileSizeBytes: int(modelBytes), LoadTime: time.Since(t0), OutOfCore: options.OutOfCore}, nil
		}
		if mmap.IsMapped() {
			prefaultMappedModel(mmap.Bytes(), header, options)
		}
	} else if mmap.IsMapped() {
		// A malformed model will fail in the real parser below. Preserve the
		// legacy full warm-up behavior when no trustworthy tensor map exists.
		if effectivePrefaultMode(options) == MmapPrefaultAll {
			prefaultPages(mmap.Bytes())
		}
	}
	// Quantized weights borrow sub-slices of the file buffer instead of
	// copying (multi-gigabyte models load without a second copy). The one
	// case that must copy: Metal builds where mmap fell back to os.ReadFile —
	// only an actual OS mapping may be retained by Metal with bytesNoCopy, a
	// C object must never keep a pointer into Go-managed heap memory. Without
	// Metal, borrowing from the heap buffer is plain Go slice aliasing and
	// always safe.
	r, err := runnerFromGGUFBytes(mmap.Bytes(), mmap.IsMapped() || !options.UseMetal, options)
	if err != nil {
		_ = mmap.Close()
		return nil, LoadInfo{}, err
	}
	r.mappedFile = mmap
	return r, LoadInfo{FileSizeBytes: mmap.Len(), LoadTime: time.Since(t0), OutOfCore: options.OutOfCore}, nil
}

func (r *Runner) Architecture() string      { return r.arch }
func (r *Runner) Tokenizer() *Tokenizer     { return r.tok }
func (r *Runner) GGUF() *GGUFFile           { return r.gguf }
func (r *Runner) Config() Config            { return r.config }
func (r *Runner) ModelName() (string, bool) { return r.gguf.GetString("general.name") }
func (r *Runner) OutOfCore() bool           { return r != nil && r.outOfCore }

// VisionConfig returns the paired vision encoder's hyperparameters, if one
// was loaded alongside this Runner's text decoder. ok is false for a
// text-only model (see HasVision).
func (r *Runner) VisionConfig() (cfg PixtralVisionConfig, ok bool) {
	if r == nil {
		return PixtralVisionConfig{}, false
	}
	r.visionMu.RLock()
	defer r.visionMu.RUnlock()
	if r.vision == nil {
		return PixtralVisionConfig{}, false
	}
	return r.visionConfig, true
}

func (r *Runner) Close() error {
	if r == nil {
		return nil
	}
	r.modelMu.Lock()
	defer r.modelMu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	r.genLock.Lock()
	defer r.genLock.Unlock()
	r.releaseMetalWeights()
	r.clearWeightContainers()
	// PrepareChatContext may render an image without taking genLock. Hold the
	// exclusive vision lease across both accelerator release and unmapping so
	// no image encoder can observe a torn tower or an invalid mmap range.
	r.visionMu.Lock()
	releasePixtralVisionWeights(r.vision)
	r.vision = nil
	r.visionConfig = PixtralVisionConfig{}
	r.visionCacheReset()
	if r.visionMappedFile != nil {
		_ = r.visionMappedFile.Close()
		r.visionMappedFile = nil
	}
	r.visionMu.Unlock()
	r.workspaceCache = nil
	r.workspaceBuf = nil
	r.bertScratch = bertEmbeddingScratch{}
	r.prefixCache = prefixCacheState{}
	r.DisableMistralPromptPrefixCache()
	// Close every shard of an out-of-core split model, keeping the first
	// error but never leaving a mapping behind: on Windows a live mapping
	// keeps the file locked.
	var err error
	for _, m := range r.extraMappedFiles {
		if m == nil {
			continue
		}
		if cerr := m.Close(); cerr != nil && err == nil {
			err = cerr
		}
	}
	r.extraMappedFiles = nil
	if r.mappedFile == nil {
		return err
	}
	if cerr := r.mappedFile.Close(); cerr != nil && err == nil {
		err = cerr
	}
	r.mappedFile = nil
	return err
}

// acquireModelLease keeps mapped and accelerator-backed weights alive for a
// resource-consuming operation. Its caller must invoke releaseModelLease.
func (r *Runner) acquireModelLease() error {
	r.modelMu.RLock()
	if r.closed {
		r.modelMu.RUnlock()
		return ErrRunnerClosed
	}
	return nil
}

func (r *Runner) releaseModelLease() { r.modelMu.RUnlock() }

func (r *Runner) releaseMetalWeights() {
	if r == nil {
		return
	}
	switch r.kind {
	case loadedBERT:
		releaseBERTMetalWeights(&r.bert)
	case loadedGptOss:
		releaseModelMetalWeights(&r.gptOss.Standard)
	case loadedGemma4:
		releaseGemma4MetalWeights(&r.gemma4)
	case loadedNemotronH:
		releaseNemotronHMetalWeights(&r.nemotronH)
	case loadedMamba2:
		releaseMamba2MetalWeights(&r.mamba2)
	case loadedQwen35:
		releaseQwen35MetalWeights(&r.qwen35)
	default:
		releaseModelMetalWeights(&r.standard)
	}
}

// clearWeightContainers drops every CPU-side tensor reference after the
// accelerator handles have been released. This matters for byte-backed
// models: individual Raw views otherwise keep the caller's entire GGUF slice
// alive after Close. The Runner contract already forbids use after Close.
func (r *Runner) clearWeightContainers() {
	if r == nil {
		return
	}
	r.standard = ModelWeights{}
	r.gptOss = GptOssWeights{}
	r.gemma4 = Gemma4Weights{}
	r.nemotronH = NemotronHWeights{}
	r.mamba2 = Mamba2Weights{}
	r.bert = BERTWeights{}
	r.qwen35 = Qwen35Weights{}
}
