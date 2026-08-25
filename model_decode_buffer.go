package gopherllm

// DecodeBuffer is reusable request scratch for single-token decode and batched
// prefill: activation vectors (X residual stream, XN/XN2 normed views,
// Q/K/V/AttnOut/Proj attention buffers, Gate/Up/Hidden FFN buffers), the
// output logits and generation/token scratch, the sampler's candidate scratch,
// and precomputed RoPE tables (per-pair inverse frequencies plus per-position
// sin/cos filled in prepareRopeScratch). One DecodeBuffer serves successive
// requests, so decode allocates nothing per token and prefill reuses its
// activation slabs. Not safe for concurrent use; Runner.genLock serializes
// requests.
type DecodeBuffer struct {
	X []float32
	// PosEmbd is scratch space for the gathered absolute-position-embedding
	// row (GPT-2/StarCoder v1 only; see Config.usesAbsolutePositionEmbd).
	PosEmbd  []float32
	XN       []float32
	XN2      []float32
	Q        []float32
	K        []float32
	V        []float32
	QKV      []float32
	AttnOut  []float32
	Proj     []float32
	AttnProj []float32
	Gate     []float32
	Up       []float32
	GateUp   []float32
	Hidden   []float32
	MOE      []float32
	// MoELatent keeps Nemotron-H's optional projected expert input alive
	// while each selected expert reuses MOE for its output.
	MoELatent []float32
	// MLA scratch holds the Q LoRA intermediate, the compact KV projection,
	// and a temporary RoPE/value expansion plane.  It is reused across all
	// layers so Kimi/DeepSeek decode remains allocation-free.
	MLAQ            []float32
	MLAKV           []float32
	MLATmp          []float32
	MLAValues       []float32
	RouterLogits    []float32
	RouterSelection []float32
	// RouterGroups/TopGroups are DeepSeek-V3's allocation-free scratch for
	// group-limited noaux routing. They remain small (eight groups in V3)
	// while TopExperts retains the final selected expert indices.
	RouterGroups            []float32
	TopGroups               []ExpertScore
	TopExperts              []ExpertScore
	ExpertProbs             []float32
	SamplerCandidates       []TokenProb
	Logits                  []float32
	RecentTokens            []uint32
	GeneratedTokens         []uint32
	StreamBytes             []byte
	Q4KXSums                []float32
	RopeInvFreq             []float32
	RopeSin                 []float32
	RopeCos                 []float32
	RopeMscale              float32
	RopeSWAInvFreq          []float32
	RopeSWASin              []float32
	RopeSWACos              []float32
	RopeSWAMscale           float32
	RopeGptOssInvFreq       []float32
	RopeGptOssConcentration float32
	MambaIn                 []float32
	MambaConv               []float32
	MambaZ                  []float32
	MambaX                  []float32
	MambaB                  []float32
	MambaC                  []float32
	MambaDT                 []float32
	MambaY                  []float32
	MambaKernel             []float32
	// MambaBeta/MambaRecall are Qwen3.5 Gated DeltaNet scratch: MambaBeta is
	// the per-head delta-rule mixing gate (reuses the Mamba naming scheme
	// rather than adding a parallel "DeltaNet*" set, since they play the same
	// per-head-scratch role as MambaDT etc.), MambaRecall holds one distinct
	// head-width span per head for what the state predicts for this token's key
	// before the delta correction. This permits the independent head updates to
	// run in parallel without per-token allocations.
	MambaBeta   []float32
	MambaRecall []float32
	// QGate/AttnGate are Qwen3.5's gated-attention scratch: attn_q projects
	// each head to [query(headDim) | gate(headDim)] rather than plain
	// headDim, so QGate holds the raw strided projection and AttnGate the
	// extracted, compactly-packed gate half (the query half is copied into
	// the ordinary Q buffer so every existing per-head helper still applies).
	QGate    []float32
	AttnGate []float32
	// MTPInput is Qwen's NextN [normalized token embedding | normalized
	// previous target hidden] pair. It is only touched by the optional Qwen
	// draft head, but keeping it in DecodeBuffer makes the draft loop
	// allocation-free once its separate workspace has been created.
	MTPInput     []float32
	ExpertRow    []float32
	ExpertHidden []float32
	// Gemma4PLE and Gemma4PLEInput retain a token's prepared E2B per-layer
	// embedding slices and their raw embedding-row source. They are grown only
	// when a native PLE model is used, so ordinary decoder workspaces stay the
	// same size.
	Gemma4PLE      []float32
	Gemma4PLEInput []float32
	batch          batchDecodeBuffer
	// ImageEmbeds maps an absolute sequence position to a vision-projector
	// embedding that must overwrite the ordinary token-embedding-table
	// lookup at that position (see runtime.go's image-placeholder token
	// splicing). Set once per generation call, cleared afterward so it can
	// never leak into an unrelated later request reusing this pooled buffer.
	ImageEmbeds map[int][]float32
}

type ExpertScore struct {
	Index int
	Score float32
}

func NewDecodeBuffer(config Config, maxHeadDim, maxNKVHeads, maxValueDim int) *DecodeBuffer {
	inv, mscale := buildRopeInvFreq(config, maxHeadDim)
	var swaInv []float32
	var swaSin, swaCos []float32
	swaMscale := float32(1)
	if config.RopeThetaSWA > 0 {
		swaConfig := config
		swaConfig.RopeTheta = config.RopeThetaSWA
		swaConfig.RopeScalingType = ""
		swaConfig.RopeScalingFactor = 1
		swaConfig.RopeOriginalContextLength = 0
		swaConfig.RopeFactorsLong = nil
		swaConfig.RopeFactorsShort = nil
		swaInv, swaMscale = buildRopeInvFreq(swaConfig, maxHeadDim)
		swaSin = make([]float32, max(1, maxHeadDim/2))
		swaCos = make([]float32, max(1, maxHeadDim/2))
	}
	gptInv, concentration := buildRopeInvFreqGptOss(config)
	return &DecodeBuffer{
		X:                       make([]float32, config.Dim),
		XN:                      make([]float32, config.Dim),
		XN2:                     make([]float32, config.Dim),
		Q:                       make([]float32, config.NHeads*maxHeadDim),
		K:                       make([]float32, maxNKVHeads*maxHeadDim),
		V:                       make([]float32, maxNKVHeads*maxValueDim),
		QKV:                     make([]float32, config.NHeads*maxHeadDim+maxNKVHeads*maxHeadDim+maxNKVHeads*maxValueDim),
		AttnOut:                 make([]float32, config.NHeads*maxValueDim),
		Proj:                    make([]float32, config.Dim),
		AttnProj:                make([]float32, config.Dim),
		Gate:                    make([]float32, config.HiddenDim),
		Up:                      make([]float32, config.HiddenDim),
		GateUp:                  make([]float32, config.HiddenDim*2),
		Hidden:                  make([]float32, config.HiddenDim),
		MOE:                     make([]float32, config.Dim),
		MoELatent:               make([]float32, config.Dim),
		MLAQ:                    make([]float32, max(1, config.MLAQueryLoRARank)),
		MLAKV:                   make([]float32, max(1, config.MLAKVLoRARank+config.RopeDimensionCount)),
		MLATmp:                  make([]float32, max(1, max(config.MLAKVLoRARank, config.RopeDimensionCount))),
		MLAValues:               make([]float32, max(1, config.NHeads*config.MLAValueDim)),
		RouterLogits:            make([]float32, config.ExpertCount),
		RouterSelection:         make([]float32, config.ExpertCount),
		RouterGroups:            make([]float32, max(1, config.ExpertGroupCount)),
		TopGroups:               make([]ExpertScore, 0, max(1, config.ExpertGroupUsedCount)),
		TopExperts:              make([]ExpertScore, 0, config.ExpertUsedCount),
		ExpertProbs:             make([]float32, 0, config.ExpertUsedCount),
		SamplerCandidates:       make([]TokenProb, 0, 64),
		Logits:                  make([]float32, config.VocabSize),
		RecentTokens:            make([]uint32, 0, repeatPenaltyWindow),
		GeneratedTokens:         make([]uint32, 0, 64),
		StreamBytes:             make([]byte, 0, 256),
		Q4KXSums:                make([]float32, max(1, config.Dim/32)),
		MTPInput:                make([]float32, 2*config.Dim),
		RopeInvFreq:             inv,
		RopeSin:                 make([]float32, max(1, maxHeadDim/2)),
		RopeCos:                 make([]float32, max(1, maxHeadDim/2)),
		RopeMscale:              mscale,
		RopeSWAInvFreq:          swaInv,
		RopeSWASin:              swaSin,
		RopeSWACos:              swaCos,
		RopeSWAMscale:           swaMscale,
		RopeGptOssInvFreq:       gptInv,
		RopeGptOssConcentration: concentration,
	}
}
