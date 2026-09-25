package gopherllm

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"

	internaltok "github.com/SimonWaldherr/GopherLLM/internal/tokenizer"
)

type layaConfig struct {
	Encoder              string             `json:"encoder"`
	HeadLayers           int                `json:"head_layers"`
	MaxLen               int                `json:"max_len"`
	HeadMaxLen           int                `json:"head_max_len"`
	Temperature          []float64          `json:"temperature"`
	TemperatureByOptions map[string]float64 `json:"temperature_by_options"`
}
type layaEncoderConfig struct {
	ModelType   string   `json:"model_type"`
	Dim         int      `json:"hidden_size"`
	Hidden      int      `json:"intermediate_size"`
	Layers      int      `json:"num_hidden_layers"`
	Heads       int      `json:"num_attention_heads"`
	Vocab       int      `json:"vocab_size"`
	MaxPosition int      `json:"max_position_embeddings"`
	Epsilon     float32  `json:"norm_eps"`
	GlobalEvery int      `json:"global_attn_every_n_layers"`
	Local       int      `json:"local_attention"`
	LayerTypes  []string `json:"layer_types"`
	GlobalTheta float64  `json:"global_rope_theta"`
	LocalTheta  float64  `json:"local_rope_theta"`
	Rope        map[string]struct {
		Theta float64 `json:"rope_theta"`
		Type  string  `json:"rope_type"`
	} `json:"rope_parameters"`
	Activation    string `json:"hidden_activation"`
	AttentionBias bool   `json:"attention_bias"`
	MLPBias       bool   `json:"mlp_bias"`
	NormBias      bool   `json:"norm_bias"`
}
type layaNorm struct {
	weight, bias []float32
	epsilon      float32
}
type layaLinear struct {
	weight Weight
	bias   []float32
}
type layaLayer struct {
	norm1, norm2          layaNorm
	qkv, output, up, down layaLinear
	theta                 float64
	window                int
	heads                 int
	gated                 bool
}

// LayaModel is a native, CPU decision encoder. It runs Laya-compatible
// ModernBERT/mmBERT checkpoints, including fine-tunes using the same graph.
// Predict is safe to share between goroutines and serializes requests. Close
// waits for an active request and releases all weights. No Python is executed.
type LayaModel struct {
	mu                            sync.Mutex
	gate                          chan struct{}
	closed                        bool
	cfg                           layaConfig
	enc                           layaEncoderConfig
	tok                           *internaltok.LayaTokenizer
	embedding                     Weight
	mapping                       *MmapFile
	embNorm, finalNorm, scoreNorm layaNorm
	layers, head                  []layaLayer
	typeEmb                       []float32
	score1, score2, act1, act2    layaLinear
}

// OpenLaya loads a local checkpoint directory containing rl_agent_config.json,
// encoder/config.json, tokenizer/{tokenizer,tokenizer_config}.json and
// model.safetensors. Use huggingface.DownloadLaya for Hub downloads.
func OpenLaya(ctx context.Context, dir string) (model *LayaModel, err error) {
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	m := &LayaModel{gate: make(chan struct{}, 1), cfg: layaConfig{HeadLayers: 2, MaxLen: 512, HeadMaxLen: 192}}
	readJSON := func(path string, out any) error {
		b, e := os.ReadFile(filepath.Join(dir, path))
		if e != nil {
			return e
		}
		return json.Unmarshal(b, out)
	}
	if err = readJSON("rl_agent_config.json", &m.cfg); err != nil {
		return nil, fmt.Errorf("laya config: %w", err)
	}
	m.enc = layaEncoderConfig{Epsilon: 1e-5, GlobalEvery: 3, Local: 128, GlobalTheta: 160000, LocalTheta: 10000, MaxPosition: 8192, Activation: "gelu"}
	if err = readJSON("encoder/config.json", &m.enc); err != nil {
		return nil, fmt.Errorf("laya encoder: %w", err)
	}
	c := &m.enc
	if c.ModelType != "modernbert" || c.Activation != "gelu" {
		return nil, fmt.Errorf("laya: unsupported encoder %q / activation %q", c.ModelType, c.Activation)
	}
	if c.Dim <= 0 || c.Dim > 8192 || c.Heads <= 0 || c.Heads > c.Dim || c.Dim%c.Heads != 0 || c.Dim/c.Heads%2 != 0 || c.Hidden <= 0 || c.Hidden > 32768 || c.Layers <= 0 || c.Layers > 128 || c.Vocab <= 0 || c.Vocab > 1000000 || c.GlobalEvery <= 0 || c.Local < 0 || c.MaxPosition <= 0 || c.MaxPosition > 131072 || c.Epsilon <= 0 {
		return nil, fmt.Errorf("laya: invalid encoder dimensions")
	}
	if m.cfg.HeadLayers < 0 || m.cfg.HeadLayers > 16 || c.Dim%max(1, c.Dim/64) != 0 || m.cfg.MaxLen < 8 || m.cfg.MaxLen > min(c.MaxPosition, 8192) || m.cfg.HeadMaxLen < 8 || m.cfg.HeadMaxLen > m.cfg.MaxLen-4 {
		return nil, fmt.Errorf("laya: invalid decision head or token budget")
	}
	if len(c.LayerTypes) != 0 && len(c.LayerTypes) != c.Layers {
		return nil, fmt.Errorf("laya: layer_types length differs from num_hidden_layers")
	}
	for k, p := range c.Rope {
		if p.Type != "" && p.Type != "default" || p.Theta <= 0 {
			return nil, fmt.Errorf("laya: unsupported RoPE parameters for %s", k)
		}
		switch k {
		case "full_attention":
			c.GlobalTheta = p.Theta
		case "sliding_attention":
			c.LocalTheta = p.Theta
		default:
			return nil, fmt.Errorf("laya: unknown attention type %q", k)
		}
	}
	if c.GlobalTheta <= 0 || c.LocalTheta <= 0 {
		return nil, fmt.Errorf("laya: invalid RoPE theta")
	}
	if len(m.cfg.Temperature) == 0 {
		m.cfg.Temperature = []float64{1, 1, 1}
	}
	if len(m.cfg.Temperature) != 3 {
		return nil, fmt.Errorf("laya: temperature needs three values")
	}
	for i, t := range m.cfg.Temperature {
		m.cfg.Temperature[i] = layaTemperature(t)
	}
	for k, t := range m.cfg.TemperatureByOptions {
		m.cfg.TemperatureByOptions[k] = layaTemperature(t)
	}
	tb, e := os.ReadFile(filepath.Join(dir, "tokenizer/tokenizer.json"))
	if e != nil {
		return nil, e
	}
	tc, e := os.ReadFile(filepath.Join(dir, "tokenizer/tokenizer_config.json"))
	if e != nil {
		return nil, e
	}
	if m.tok, err = internaltok.NewLayaTokenizer(tb, tc); err != nil {
		return nil, err
	}
	if err = m.tok.ValidateVocab(c.Vocab); err != nil {
		return nil, err
	}
	if m.mapping, err = OpenMmap(filepath.Join(dir, "model.safetensors")); err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			m.mapping.Close()
		}
	}()
	sf, e := ParseSafetensors(m.mapping.Bytes())
	if e != nil {
		return nil, e
	}
	// Validate every shape and byte range before any tensor-sized allocation.
	for name, ti := range sf.Tensors {
		n := int64(1)
		for _, d := range ti.Shape {
			if d <= 0 || n > math.MaxInt32/int64(d) {
				return nil, fmt.Errorf("laya: invalid tensor shape %s", name)
			}
			n *= int64(d)
		}
		size := int64(2)
		if ti.DType == SafetensorsF32 {
			size = 4
		} else if ti.DType != SafetensorsF16 && ti.DType != SafetensorsBF16 {
			return nil, fmt.Errorf("laya: unsupported dtype %s", ti.DType)
		}
		if ti.Begin < 0 || ti.End < ti.Begin || ti.End > len(sf.Data)-sf.DataOffset || int64(ti.End-ti.Begin) != n*size {
			return nil, fmt.Errorf("laya: invalid tensor range %s", name)
		}
	}
	used := map[string]bool{"encoder.embeddings.tok_embeddings.weight": true}
	var loadErr error
	tensor := func(name string, shape ...int) []float32 {
		if loadErr != nil {
			return nil
		}
		if e := ctx.Err(); e != nil {
			loadErr = e
			return nil
		}
		used[name] = true
		info, ok := sf.Tensors[name]
		if !ok || !slices.Equal(info.Shape, shape) {
			loadErr = fmt.Errorf("laya: tensor %s has shape %v, expected %v", name, info.Shape, shape)
			return nil
		}
		var v []float32
		v, loadErr = sf.F32(name)
		return v
	}
	norm := func(name string, bias bool, eps float32) layaNorm {
		n := layaNorm{weight: tensor(name+".weight", c.Dim), epsilon: eps}
		if bias {
			n.bias = tensor(name+".bias", c.Dim)
		}
		return n
	}
	linear := func(name string, rows, cols int, bias bool) layaLinear {
		l := layaLinear{weight: Weight{F32: tensor(name+".weight", rows, cols), Rows: rows, Cols: cols}}
		if bias {
			l.bias = tensor(name+".bias", rows)
		}
		return l
	}
	ei, ok := sf.Tensors["encoder.embeddings.tok_embeddings.weight"]
	if !ok || !slices.Equal(ei.Shape, []int{c.Vocab, c.Dim}) {
		return nil, fmt.Errorf("laya: invalid embedding shape")
	}
	typ := GGMLTypeF16
	if ei.DType == SafetensorsF32 {
		typ = GGMLTypeF32
	} else if ei.DType == SafetensorsBF16 {
		typ = GGMLTypeBF16
	}
	m.embedding = Weight{Raw: sf.Data[sf.DataOffset+ei.Begin : sf.DataOffset+ei.End], Type: typ, Rows: c.Vocab, Cols: c.Dim}
	m.embNorm = norm("encoder.embeddings.norm", c.NormBias, c.Epsilon)
	m.finalNorm = norm("encoder.final_norm", c.NormBias, c.Epsilon)
	for i := 0; i < c.Layers; i++ {
		prefix := fmt.Sprintf("encoder.layers.%d.", i)
		l := layaLayer{heads: c.Heads, gated: true, window: -1, theta: c.GlobalTheta}
		local := i%c.GlobalEvery != 0
		if len(c.LayerTypes) > 0 {
			switch c.LayerTypes[i] {
			case "full_attention":
				local = false
			case "sliding_attention":
				local = true
			default:
				return nil, fmt.Errorf("laya: unsupported layer type %q", c.LayerTypes[i])
			}
		}
		if local {
			l.window = c.Local / 2
			l.theta = c.LocalTheta
		}
		if i > 0 {
			l.norm1 = norm(prefix+"attn_norm", c.NormBias, c.Epsilon)
		}
		l.norm2 = norm(prefix+"mlp_norm", c.NormBias, c.Epsilon)
		l.qkv = linear(prefix+"attn.Wqkv", 3*c.Dim, c.Dim, c.AttentionBias)
		l.output = linear(prefix+"attn.Wo", c.Dim, c.Dim, c.AttentionBias)
		l.up = linear(prefix+"mlp.Wi", 2*c.Hidden, c.Dim, c.MLPBias)
		l.down = linear(prefix+"mlp.Wo", c.Dim, c.Hidden, c.MLPBias)
		m.layers = append(m.layers, l)
	}
	m.typeEmb = tensor("type_emb.weight", 3, c.Dim)
	_ = tensor("temperature", 3) // Persistent model buffer; runtime calibration uses the agent config.
	for i := 0; i < m.cfg.HeadLayers; i++ {
		p := fmt.Sprintf("head.layers.%d.", i)
		l := layaLayer{heads: max(1, c.Dim/64), window: -1}
		l.norm1 = norm(p+"norm1", true, 1e-5)
		l.norm2 = norm(p+"norm2", true, 1e-5)
		l.qkv = layaLinear{weight: Weight{F32: tensor(p+"self_attn.in_proj_weight", 3*c.Dim, c.Dim), Rows: 3 * c.Dim, Cols: c.Dim}, bias: tensor(p+"self_attn.in_proj_bias", 3*c.Dim)}
		l.output = linear(p+"self_attn.out_proj", c.Dim, c.Dim, true)
		l.up = linear(p+"linear1", 4*c.Dim, c.Dim, true)
		l.down = linear(p+"linear2", c.Dim, 4*c.Dim, true)
		m.head = append(m.head, l)
	}
	m.scoreNorm = norm("scorer.0", true, 1e-5)
	m.score1 = linear("scorer.1", c.Dim, c.Dim, true)
	m.score2 = linear("scorer.3", 1, c.Dim, true)
	m.act1 = linear("act_head.0", 256, c.Dim+4, true)
	m.act2 = linear("act_head.2", 2, 256, true)
	if loadErr != nil {
		return nil, loadErr
	}
	for name := range sf.Tensors {
		if !used[name] {
			return nil, fmt.Errorf("laya: unrecognized tensor %s (unsupported decision graph)", name)
		}
	}
	return m, nil
}
func layaTemperature(t float64) float64 {
	if math.IsNaN(t) || math.IsInf(t, 0) {
		return 1
	}
	return min(5, max(.5, t))
}
func (m *LayaModel) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	m.embedding = Weight{}
	m.layers = nil
	m.head = nil
	m.typeEmb = nil
	m.embNorm, m.finalNorm, m.scoreNorm = layaNorm{}, layaNorm{}, layaNorm{}
	m.score1 = layaLinear{}
	m.score2 = layaLinear{}
	m.act1 = layaLinear{}
	m.act2 = layaLinear{}
	m.tok = nil
	if m.mapping != nil {
		return m.mapping.Close()
	}
	return nil
}
