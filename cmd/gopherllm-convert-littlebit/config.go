package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/SimonWaldherr/GopherLLM/internal/formats/gguf"
)

type hfConfig struct {
	ModelType      string   `json:"model_type"`
	Name           string   `json:"_name_or_path"`
	Hidden         int      `json:"hidden_size"`
	Intermediate   int      `json:"intermediate_size"`
	Layers         int      `json:"num_hidden_layers"`
	Heads          int      `json:"num_attention_heads"`
	KVHeads        int      `json:"num_key_value_heads"`
	HeadDim        int      `json:"head_dim"`
	Vocab          int      `json:"vocab_size"`
	MaxPosition    int      `json:"max_position_embeddings"`
	Epsilon        float32  `json:"rms_norm_eps"`
	RopeTheta      float64  `json:"rope_theta"`
	Activation     string   `json:"hidden_act"`
	PartialRoPE    float64  `json:"partial_rotary_factor"`
	Rope           *hfRoPE  `json:"rope_scaling"`
	RopeParameters *hfRoPE  `json:"rope_parameters"`
	AttentionBias  bool     `json:"attention_bias"`
	MLPBias        bool     `json:"mlp_bias"`
	TieEmbeddings  bool     `json:"tie_word_embeddings"`
	BOS            *uint32  `json:"bos_token_id"`
	EOS            tokenIDs `json:"eos_token_id"`
	Pad            *uint32  `json:"pad_token_id"`
	SlidingWindow  bool     `json:"use_sliding_window"`
	hfQuantConfig
}

type hfQuantConfig struct {
	ITQ       *bool  `json:"use_itq"`
	Module    string `json:"quant_mod"`
	Quantizer string `json:"quant_func"`
}

type hfRoPE struct {
	Type      string  `json:"rope_type"`
	OldType   string  `json:"type"`
	Theta     float64 `json:"rope_theta"`
	Factor    float64 `json:"factor"`
	LowFreq   float64 `json:"low_freq_factor"`
	HighFreq  float64 `json:"high_freq_factor"`
	OldLength int     `json:"original_max_position_embeddings"`
}

type tokenIDs []uint32

func (t *tokenIDs) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		return nil
	}
	if b[0] == '[' {
		return json.Unmarshal(b, (*[]uint32)(t))
	}
	var id uint32
	if err := json.Unmarshal(b, &id); err != nil {
		return err
	}
	*t = []uint32{id}
	return nil
}

func loadConfig(dir string, variant uint32) (hfConfig, uint32, error) {
	var cfg hfConfig
	if err := readJSON(dir, "config.json", &cfg); err != nil {
		return cfg, 0, err
	}
	if cfg.ModelType != "llama" && cfg.ModelType != "qwen3" {
		return cfg, 0, fmt.Errorf("unsupported checkpoint model_type %q (expected llama or qwen3)", cfg.ModelType)
	}
	if cfg.SlidingWindow {
		return cfg, 0, fmt.Errorf("checkpoints with sliding-window attention are unsupported")
	}
	quant := cfg.hfQuantConfig
	if err := readJSON(dir, "littlebit_config.json", &quant); err != nil && !errors.Is(err, os.ErrNotExist) {
		return cfg, 0, err
	}
	if quant.Module != "" && quant.Module != "LittleBitLinear" {
		return cfg, 0, fmt.Errorf("unsupported LittleBit quant_mod %q", quant.Module)
	}
	if quant.Quantizer != "" && quant.Quantizer != "STEBinary" && quant.Quantizer != "SmoothSign" {
		return cfg, 0, fmt.Errorf("unsupported LittleBit quant_func %q", quant.Quantizer)
	}
	if variant == 0 {
		if quant.ITQ == nil {
			return cfg, 0, fmt.Errorf("checkpoint does not declare use_itq; pass -variant 1 or -variant 2")
		}
		variant = 1
		if *quant.ITQ {
			variant = 2
		}
	}
	if variant != 1 && variant != 2 {
		return cfg, 0, fmt.Errorf("unsupported LittleBit variant %d", variant)
	}
	if cfg.Heads <= 0 || cfg.Hidden <= 0 || cfg.Intermediate <= 0 || cfg.Layers <= 0 || cfg.Vocab <= 0 || cfg.MaxPosition <= 0 {
		return cfg, 0, fmt.Errorf("missing or invalid decoder dimensions")
	}
	if cfg.KVHeads == 0 {
		cfg.KVHeads = cfg.Heads
	}
	if cfg.HeadDim == 0 {
		if cfg.Hidden%cfg.Heads != 0 {
			return cfg, 0, fmt.Errorf("hidden_size must be divisible by num_attention_heads")
		}
		cfg.HeadDim = cfg.Hidden / cfg.Heads
	}
	if cfg.HeadDim < 2 || cfg.HeadDim%2 != 0 || cfg.KVHeads <= 0 || cfg.Heads%cfg.KVHeads != 0 ||
		cfg.Heads > math.MaxInt/cfg.HeadDim || uint64(cfg.Heads) > math.MaxUint32/uint64(cfg.HeadDim) {
		return cfg, 0, fmt.Errorf("invalid decoder attention dimensions")
	}
	for _, n := range []int{cfg.Hidden, cfg.Intermediate, cfg.Layers, cfg.MaxPosition, cfg.Vocab} {
		if uint64(n) > math.MaxUint32 {
			return cfg, 0, fmt.Errorf("decoder dimensions overflow GGUF metadata")
		}
	}
	if cfg.Activation != "" && cfg.Activation != "silu" && cfg.Activation != "swish" {
		return cfg, 0, fmt.Errorf("unsupported hidden_act %q", cfg.Activation)
	}
	if cfg.Epsilon == 0 {
		cfg.Epsilon = 1e-6
	}
	if cfg.RopeParameters != nil {
		cfg.Rope = cfg.RopeParameters
	}
	if cfg.Rope != nil && cfg.Rope.Theta != 0 {
		cfg.RopeTheta = cfg.Rope.Theta
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000
	}
	if cfg.RopeTheta <= 0 || cfg.RopeTheta > math.MaxFloat32 || cfg.Epsilon <= 0 ||
		math.IsInf(float64(cfg.Epsilon), 0) || cfg.PartialRoPE != 0 && cfg.PartialRoPE != 1 {
		return cfg, 0, fmt.Errorf("invalid RoPE or RMSNorm configuration (partial RoPE is unsupported)")
	}
	return cfg, variant, nil
}

func (cfg hfConfig) metadata(dir string) (map[string]gguf.MetaValue, error) {
	name := cfg.Name
	if name == "" {
		name = filepath.Base(dir)
	}
	meta := map[string]gguf.MetaValue{
		"general.architecture": {Kind: "str", Value: cfg.ModelType},
		"general.name":         {Kind: "str", Value: name},
		cfg.ModelType + ".attention.layer_norm_rms_epsilon": {Kind: "f32", Value: cfg.Epsilon},
		cfg.ModelType + ".rope.freq_base":                   {Kind: "f32", Value: float32(cfg.RopeTheta)},
	}
	for k, v := range map[string]int{
		"context_length": cfg.MaxPosition, "embedding_length": cfg.Hidden, "feed_forward_length": cfg.Intermediate,
		"block_count": cfg.Layers, "vocab_size": cfg.Vocab, "attention.head_count": cfg.Heads, "attention.head_count_kv": cfg.KVHeads,
		"attention.key_length": cfg.HeadDim, "attention.value_length": cfg.HeadDim, "rope.dimension_count": cfg.HeadDim,
	} {
		meta[cfg.ModelType+"."+k] = gguf.MetaValue{Kind: "u32", Value: uint32(v)}
	}
	if err := cfg.tokenizer(dir, meta); err != nil {
		return nil, err
	}
	return meta, nil
}

func (cfg hfConfig) ropeFactors() ([]byte, error) {
	rope := cfg.Rope
	if rope == nil {
		return nil, nil
	}
	kind := rope.Type
	if kind == "" {
		kind = rope.OldType
	}
	switch kind {
	case "default":
		return nil, nil
	case "llama3":
		if rope.Factor < 1 || rope.OldLength <= 0 || rope.LowFreq <= 0 || rope.HighFreq <= rope.LowFreq {
			return nil, fmt.Errorf("invalid llama3 RoPE scaling")
		}
	case "linear":
		if rope.Factor < 1 {
			return nil, fmt.Errorf("invalid linear RoPE scaling")
		}
	default:
		return nil, fmt.Errorf("unsupported RoPE scaling %q", kind)
	}
	factors := make([]byte, cfg.HeadDim*2)
	for i := range cfg.HeadDim / 2 {
		factor := rope.Factor
		if kind == "llama3" {
			wavelength := 2 * math.Pi * math.Pow(cfg.RopeTheta, float64(2*i)/float64(cfg.HeadDim))
			smooth := (float64(rope.OldLength)/wavelength - rope.LowFreq) / (rope.HighFreq - rope.LowFreq)
			smooth = min(max(smooth, 0), 1)
			factor = 1 / (smooth + (1-smooth)/factor)
		}
		if factor > math.MaxFloat32 {
			return nil, fmt.Errorf("RoPE frequency factor overflows")
		}
		binary.LittleEndian.PutUint32(factors[i*4:], math.Float32bits(float32(factor)))
	}
	return factors, nil
}
