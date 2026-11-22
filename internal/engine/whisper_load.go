package engine

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
)

// WhisperModel is a local, text-only Whisper encoder-decoder. Model tensors
// stored as F16 borrow the read-only checkpoint mapping.
type WhisperModel struct {
	mapping                        *MmapFile
	audioCtx, audioDim, audioHeads int
	textCtx, textDim, textHeads    int
	mels, vocab                    int
	encoder, decoder               []whisperBlock
	conv1, conv2                   whisperLinear
	audioNorm, textNorm            whisperNorm
	audioPos, textPos              []float32
	embedding                      Weight
	filterbank                     Weight
	pieces                         []string
	suppress                       []bool
	end, start, languages          int
}

type whisperNorm struct{ weight, bias []float32 }

type whisperLinear struct {
	weight Weight
	bias   []float32
}

type whisperAttention struct{ query, key, value, out whisperLinear }

type whisperBlock struct {
	self, cross                  whisperAttention
	selfNorm, crossNorm, mlpNorm whisperNorm
	expand, contract             whisperLinear
}

type whisperTensor struct {
	dims []int
	raw  []byte
	typ  GGMLType
}

type whisperReader struct {
	data []byte
	err  error
}

func (r *whisperReader) take(n int) []byte {
	if r.err != nil {
		return nil
	}
	if n < 0 || n > len(r.data) {
		r.err = fmt.Errorf("truncated Whisper checkpoint")
		return nil
	}
	b := r.data[:n:n]
	r.data = r.data[n:]
	return b
}

func (r *whisperReader) integer() int {
	b := r.take(4)
	if b == nil {
		return 0
	}
	return int(int32(binary.LittleEndian.Uint32(b)))
}

// OpenWhisper loads a local legacy GGML Whisper checkpoint with F32/F16 weights.
// The container is parsed by Go. No external inference runtime is required.
func OpenWhisper(ctx context.Context, path, language string) (*WhisperModel, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	mapping, err := OpenMmap(path)
	if err != nil {
		return nil, fmt.Errorf("Whisper: %w", err)
	}
	model, err := parseWhisper(ctx, mapping.Bytes())
	if err == nil {
		_, err = model.whisperLanguage(language)
	}
	if err != nil {
		_ = mapping.Close()
		return nil, fmt.Errorf("Whisper: %w", err)
	}
	model.mapping = mapping
	return model, nil
}

func parseWhisper(ctx context.Context, data []byte) (*WhisperModel, error) {
	r := whisperReader{data: data}
	if r.integer() != 0x67676d6c {
		return nil, fmt.Errorf("expected a legacy GGML Whisper checkpoint")
	}
	m := &WhisperModel{}
	m.vocab = r.integer()
	m.audioCtx, m.audioDim, m.audioHeads = r.integer(), r.integer(), r.integer()
	audioLayers := r.integer()
	m.textCtx, m.textDim, m.textHeads = r.integer(), r.integer(), r.integer()
	textLayers := r.integer()
	m.mels = r.integer()
	fileType := r.integer()
	if r.err != nil {
		return nil, r.err
	}
	if fileType != 0 && fileType != 1 {
		return nil, fmt.Errorf("unsupported checkpoint precision %d (use F16 or F32)", fileType)
	}
	if m.vocab < 51864 || m.vocab > 51866 || m.audioCtx != 1500 || m.textCtx != 448 ||
		(m.mels != 80 && m.mels != 128) || !validWhisperWidth(m.audioDim, m.audioHeads) ||
		!validWhisperWidth(m.textDim, m.textHeads) || m.audioDim != m.textDim ||
		audioLayers < 1 || audioLayers > 32 || textLayers < 1 || textLayers > 32 {
		return nil, fmt.Errorf("unsupported encoder, decoder or vocabulary dimensions")
	}
	m.end = 50257
	m.languages = m.vocab - 51766
	if m.vocab == 51864 {
		m.end = 50256
		m.languages = 99
	}
	m.start = m.end + 1
	mels, bins := r.integer(), r.integer()
	if mels != m.mels || bins != 201 {
		return nil, fmt.Errorf("invalid mel filterbank shape")
	}
	filterData := r.take(mels * bins * 4)
	pieceCount := r.integer()
	if pieceCount < m.end || pieceCount > m.vocab {
		return nil, fmt.Errorf("invalid vocabulary size %d", pieceCount)
	}
	m.pieces, m.suppress = make([]string, m.end), make([]bool, m.end)
	for i := range pieceCount {
		size := r.integer()
		if size < 0 || size > 4096 {
			return nil, fmt.Errorf("invalid vocabulary entry length")
		}
		piece := r.take(size)
		if r.err != nil {
			return nil, r.err
		}
		if i < m.end {
			m.pieces[i] = string(piece)
			m.suppress[i] = whisperSuppressPiece(m.pieces[i])
		}
	}
	m.filterbank = Weight{Cols: bins, F32: whisperFloat32(filterData)}
	tensors := make(map[string]whisperTensor)
	for len(r.data) > 0 && r.err == nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		rank, nameLen, kind := r.integer(), r.integer(), r.integer()
		if rank < 1 || rank > 3 || nameLen < 1 || nameLen > 256 || (kind != 0 && kind != 1) || len(tensors) >= 4096 {
			return nil, fmt.Errorf("invalid or unsupported tensor descriptor")
		}
		dims, n := make([]int, rank), 1
		for i := range dims {
			dims[i] = r.integer()
			if dims[i] < 1 || dims[i] > 1<<26/n {
				return nil, fmt.Errorf("tensor dimensions are invalid or exceed the Whisper limit")
			}
			n *= dims[i]
		}
		name := string(r.take(nameLen))
		raw := r.take(n * (4 >> kind))
		if _, duplicate := tensors[name]; duplicate {
			return nil, fmt.Errorf("duplicate tensor %s", name)
		}
		tensors[name] = whisperTensor{dims: dims, raw: raw, typ: GGMLType(kind)}
	}
	if r.err != nil {
		return nil, r.err
	}
	if err := m.loadWhisperWeights(tensors, audioLayers, textLayers); err != nil {
		return nil, err
	}
	return m, ctx.Err()
}

// Width and head limits match the public Whisper family. Each head has 64 channels.
func validWhisperWidth(width, heads int) bool {
	return heads >= 1 && heads <= 20 && width == heads*64
}

func whisperFloat32(raw []byte) []float32 {
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

func (m *WhisperModel) loadWhisperWeights(tensors map[string]whisperTensor, nAudio, nText int) error {
	var err error
	load := func(name string, dims ...int) Weight {
		if err != nil {
			return Weight{}
		}
		t, ok := tensors[name]
		if !ok {
			err = fmt.Errorf("missing tensor %s", name)
			return Weight{}
		}
		if !equalWhisperDims(t.dims, dims) {
			err = fmt.Errorf("tensor %s has shape %v, want %v", name, t.dims, dims)
			return Weight{}
		}
		rows := dims[len(dims)-1]
		size := 4
		if t.typ == GGMLTypeF16 {
			size = 2
		}
		w := Weight{Raw: t.raw, Type: t.typ, Rows: rows, Cols: len(t.raw) / size / rows}
		if t.typ == GGMLTypeF32 {
			w.F32 = whisperFloat32(t.raw)
			w.Raw = nil
		}
		return w
	}
	owned := func(w Weight) []float32 {
		if w.F32 != nil || w.Raw == nil {
			return w.F32
		}
		values, _ := dequantTensor(w.Type, w.Raw, len(w.Raw)/2)
		return values
	}
	norm := func(prefix string, dim int) whisperNorm {
		return whisperNorm{owned(load(prefix+".weight", dim)), owned(load(prefix+".bias", dim))}
	}
	linear := func(prefix string, input, output int, bias bool) whisperLinear {
		l := whisperLinear{weight: load(prefix+".weight", input, output)}
		if bias {
			l.bias = owned(load(prefix+".bias", output))
		}
		return l
	}
	attention := func(prefix string, dim int) whisperAttention {
		return whisperAttention{
			query: linear(prefix+".query", dim, dim, true),
			key:   linear(prefix+".key", dim, dim, false),
			value: linear(prefix+".value", dim, dim, true),
			out:   linear(prefix+".out", dim, dim, true),
		}
	}
	conv := func(prefix string, in, out int) whisperLinear {
		return whisperLinear{load(prefix+".weight", 3, in, out), owned(load(prefix+".bias", out))}
	}
	layer := func(prefix string, dim int, cross bool) whisperBlock {
		b := whisperBlock{
			self:     attention(prefix+".attn", dim),
			selfNorm: norm(prefix+".attn_ln", dim), mlpNorm: norm(prefix+".mlp_ln", dim),
			expand:   linear(prefix+".mlp.0", dim, 4*dim, true),
			contract: linear(prefix+".mlp.2", 4*dim, dim, true),
		}
		if cross {
			b.cross = attention(prefix+".cross_attn", dim)
			b.crossNorm = norm(prefix+".cross_attn_ln", dim)
		}
		return b
	}
	m.conv1, m.conv2 = conv("encoder.conv1", m.mels, m.audioDim), conv("encoder.conv2", m.audioDim, m.audioDim)
	m.audioPos = owned(load("encoder.positional_embedding", m.audioDim, m.audioCtx))
	m.textPos = owned(load("decoder.positional_embedding", m.textDim, m.textCtx))
	m.audioNorm, m.textNorm = norm("encoder.ln_post", m.audioDim), norm("decoder.ln", m.textDim)
	m.embedding = load("decoder.token_embedding.weight", m.textDim, m.vocab)
	m.encoder, m.decoder = make([]whisperBlock, nAudio), make([]whisperBlock, nText)
	for i := range m.encoder {
		m.encoder[i] = layer(fmt.Sprintf("encoder.blocks.%d", i), m.audioDim, false)
	}
	for i := range m.decoder {
		m.decoder[i] = layer(fmt.Sprintf("decoder.blocks.%d", i), m.textDim, true)
	}
	return err
}

func equalWhisperDims(got, want []int) bool {
	if len(want) == 1 && len(got) == 2 && got[0] == 1 {
		got = got[1:]
	}
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// Close releases the model mapping. The owner must wait for active calls.
func (m *WhisperModel) Close() error {
	if m.mapping == nil {
		return nil
	}
	err := m.mapping.Close()
	m.mapping = nil
	return err
}
