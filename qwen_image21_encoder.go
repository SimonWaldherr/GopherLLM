package gopherllm

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	tokenizerpkg "github.com/SimonWaldherr/GopherLLM/internal/tokenizer"
)

const qi21SystemPrefix = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"

// Text-only Qwen3-VL uses identical positions on its three rotary axes, so
// mRoPE reduces to the standard Qwen3 split-half rotation. Vision/DeepStack
// conditioning is deliberately not part of this adapter.
type qi21TextEncoder struct {
	config    Config
	weights   ModelWeights
	tokenizer *tokenizerpkg.QwenImageTokenizer
	drop      int
}

func newQI21TextEncoder(configJSON, tokenizerJSON []byte, source qi21WeightSource) (*qi21TextEncoder, error) {
	return newQwenVLTextEncoder(configJSON, tokenizerJSON, source, 0)
}

// layerLimit permits image models that consume intermediate hidden states to
// omit the unused tail. Zero loads all layers, as required by Qwen-Image-2.1.
func newQwenVLTextEncoder(configJSON, tokenizerJSON []byte, source qi21WeightSource, layerLimit int) (*qi21TextEncoder, error) {
	var raw struct {
		ModelType string `json:"model_type"`
		Text      struct {
			ModelType     string  `json:"model_type"`
			Dim           int     `json:"hidden_size"`
			Hidden        int     `json:"intermediate_size"`
			Layers        int     `json:"num_hidden_layers"`
			Heads         int     `json:"num_attention_heads"`
			KVHeads       int     `json:"num_key_value_heads"`
			HeadDim       int     `json:"head_dim"`
			Vocab         int     `json:"vocab_size"`
			MaxPositions  int     `json:"max_position_embeddings"`
			Epsilon       float32 `json:"rms_norm_eps"`
			Theta         float32 `json:"rope_theta"`
			Activation    string  `json:"hidden_act"`
			AttentionBias bool    `json:"attention_bias"`
			Rope          struct {
				Type string `json:"rope_type"`
			} `json:"rope_scaling"`
		} `json:"text_config"`
	}
	if len(configJSON) > 1<<20 {
		return nil, fmt.Errorf("qwen image encoder: configuration exceeds 1 MiB")
	}
	if err := json.Unmarshal(configJSON, &raw); err != nil {
		return nil, fmt.Errorf("qwen image encoder: %w", err)
	}
	c := raw.Text
	if raw.ModelType != "qwen3_vl" || c.ModelType != "qwen3_vl_text" || c.Activation != "silu" || c.AttentionBias || c.Rope.Type != "default" {
		return nil, fmt.Errorf("qwen image encoder: unsupported Qwen3-VL text configuration")
	}
	if c.Dim < 1 || c.Dim > 16384 || c.Hidden < 1 || c.Hidden > 131072 || c.Layers < 1 || c.Layers > 128 || c.Heads < 1 || c.Heads > 256 || c.KVHeads < 1 || c.Heads%c.KVHeads != 0 || c.HeadDim < 2 || c.HeadDim > 256 || c.HeadDim%2 != 0 || c.Vocab < 256 || c.Vocab > 1<<20 || c.MaxPositions < 1 || !finite32(c.Epsilon) || c.Epsilon <= 0 || !finite32(c.Theta) || c.Theta <= 0 {
		return nil, fmt.Errorf("qwen image encoder: invalid or oversized text dimensions")
	}
	if layerLimit < 0 || layerLimit > c.Layers {
		return nil, fmt.Errorf("qwen image encoder: invalid layer limit")
	}
	if layerLimit > 0 {
		c.Layers = layerLimit
	}
	tok, err := tokenizerpkg.NewQwenImageTokenizer(tokenizerJSON)
	if err != nil {
		return nil, err
	}
	if err := tok.ValidateVocab(c.Vocab); err != nil {
		return nil, err
	}
	prefix, err := tok.Encode(qi21SystemPrefix)
	if err != nil {
		return nil, err
	}
	e := &qi21TextEncoder{tokenizer: tok, drop: len(prefix), config: Config{
		Arch: "qwen3", Dim: c.Dim, HiddenDim: c.Hidden, NLayers: c.Layers, NHeads: c.Heads, NKVHeads: c.KVHeads, HeadDim: c.HeadDim, ValueDim: c.HeadDim, KVDim: c.KVHeads * c.HeadDim, KVMul: c.Heads / c.KVHeads, VocabSize: c.Vocab, MaxSeqLen: c.MaxPositions, RopeTheta: c.Theta, RopeDimensionCount: c.HeadDim, RMSNormEps: c.Epsilon, AttentionScale: float32(1 / math.Sqrt(float64(c.HeadDim))), EmbeddingScale: 1, ResidualScale: 1, LogitScale: 1,
	}}
	const base = "model.language_model."
	e.weights.TokenEmbd, err = source.matrix(base+"embed_tokens.weight", c.Vocab, c.Dim)
	if err != nil {
		return nil, err
	}
	e.weights.Layers = make([]LayerWeights, c.Layers)
	for i := range e.weights.Layers {
		l := &e.weights.Layers[i]
		p := fmt.Sprintf("%slayers.%d.", base, i)
		for _, m := range []struct {
			name       string
			rows, cols int
			dst        *Weight
		}{
			{"self_attn.q_proj.weight", c.Heads * c.HeadDim, c.Dim, &l.WQ},
			{"self_attn.k_proj.weight", c.KVHeads * c.HeadDim, c.Dim, &l.WK},
			{"self_attn.v_proj.weight", c.KVHeads * c.HeadDim, c.Dim, &l.WV},
			{"self_attn.o_proj.weight", c.Dim, c.Heads * c.HeadDim, &l.WO},
			{"mlp.gate_proj.weight", c.Hidden, c.Dim, &l.W1},
			{"mlp.up_proj.weight", c.Hidden, c.Dim, &l.W3},
			{"mlp.down_proj.weight", c.Dim, c.Hidden, &l.W2},
		} {
			*m.dst, err = source.matrix(p+m.name, m.rows, m.cols)
			if err != nil {
				return nil, err
			}
		}
		for _, v := range []struct {
			name string
			size int
			dst  *[]float32
		}{
			{"input_layernorm.weight", c.Dim, &l.AttnNorm},
			{"post_attention_layernorm.weight", c.Dim, &l.FFNNorm},
			{"self_attn.q_norm.weight", c.HeadDim, &l.AttnQNorm},
			{"self_attn.k_norm.weight", c.HeadDim, &l.AttnKNorm},
		} {
			*v.dst, err = source.floats(p+v.name, v.size)
			if err != nil {
				return nil, err
			}
		}
	}
	// No LM head, vision tensors or final RMSNorm: the DiT consumes the last
	// decoder residual, not normalized generation features.
	return e, nil
}

func (e *qi21TextEncoder) Encode(ctx context.Context, prompt string) ([]float32, int, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	if len(prompt) > 1<<20 {
		return nil, 0, fmt.Errorf("qwen image encoder: prompt exceeds 1 MiB")
	}
	if strings.Contains(prompt, "<|image_pad|>") || strings.Contains(prompt, "<|video_pad|>") {
		return nil, 0, fmt.Errorf("qwen image encoder: image/video conditioning is not supported")
	}
	if prompt == "" {
		prompt = " "
	}
	tokens, err := e.tokenizer.Encode(qi21SystemPrefix + "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n")
	if err != nil {
		return nil, 0, err
	}
	if len(tokens) <= e.drop || len(tokens) > min(4096, e.config.MaxSeqLen) {
		return nil, 0, fmt.Errorf("qwen image encoder: prompt requires %d tokens; supported limit is %d", len(tokens), min(4096, e.config.MaxSeqLen))
	}
	c := e.config
	// The official encoder needs about 1.2 GiB at the 4096-token limit.
	// Bound the combined cache before allocating it for custom configs.
	if uint64(len(tokens))*uint64(c.NLayers)*uint64(c.KVDim)*8 > 2<<30 {
		return nil, 0, fmt.Errorf("qwen image encoder: prompt KV cache exceeds 2 GiB")
	}
	cache := NewKVCache(c.NLayers, c.KVDim, c.KVDim, len(tokens))
	buf := NewDecodeBuffer(c, c.HeadDim, c.NKVHeads, c.ValueDim)
	defer func() {
		if buf.metalDense != nil {
			buf.metalDense.close()
		}
	}()
	out := make([]float32, (len(tokens)-e.drop)*c.Dim)
	for start := 0; start < len(tokens); start += 32 {
		end := min(start+32, len(tokens))
		if err := forwardBatchIntoContext(ctx, c, e.weights, cache, buf, tokens[start:end], start, false, nil, nil); err != nil {
			return nil, 0, err
		}
		// Normalizing these rows would silently change the DiT's conditioning.
		begin := max(start, e.drop)
		if begin < end {
			copy(out[(begin-e.drop)*c.Dim:(end-e.drop)*c.Dim], buf.batch.XFlat[(begin-start)*c.Dim:(end-start)*c.Dim])
		}
	}
	for _, v := range out {
		if !finite32(v) {
			return nil, 0, fmt.Errorf("qwen image encoder: non-finite prompt features")
		}
	}
	return out, len(tokens) - e.drop, nil
}
