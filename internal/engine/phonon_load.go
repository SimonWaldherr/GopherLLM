package engine

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"

	"github.com/SimonWaldherr/GopherLLM/internal/formats/phonon"
)

const maxPhononConfigBytes = 1 << 20

type phononConfig struct {
	ModelType string   `json:"model_type"`
	Labels    []string `json:"labels"`
	Fermion   struct {
		ContainerFormat string `json:"container_format"`
	} `json:"fermion"`
	Preprocessor struct {
		SampleRate    int     `json:"sample_rate"`
		Features      int     `json:"features"`
		NFFT          int     `json:"n_fft"`
		WindowSize    float64 `json:"window_size"`
		WindowStride  float64 `json:"window_stride"`
		Window        string  `json:"window"`
		Normalize     string  `json:"normalize"`
		Log           bool    `json:"log"`
		FrameSplicing int     `json:"frame_splicing"`
		PadTo         int     `json:"pad_to"`
	} `json:"preprocessor"`
	Encoder struct {
		Features            int    `json:"feat_in"`
		Layers              int    `json:"n_layers"`
		Dim                 int    `json:"d_model"`
		FFExpansion         int    `json:"ff_expansion_factor"`
		Heads               int    `json:"n_heads"`
		ConvKernel          int    `json:"conv_kernel_size"`
		Subsampling         string `json:"subsampling"`
		SubsamplingFactor   int    `json:"subsampling_factor"`
		SubsamplingChannels int    `json:"subsampling_conv_channels"`
		Attention           string `json:"self_attention_model"`
		AttentionContext    []int  `json:"att_context_size"`
		PositionLimit       int    `json:"pos_emb_max_len"`
		ConvNorm            string `json:"conv_norm_type"`
		UseBias             bool   `json:"use_bias"`
		ScaleInput          bool   `json:"xscaling"`
		CausalDownsampling  bool   `json:"causal_downsampling"`
	} `json:"encoder"`
	Decoder struct {
		VocabSize  int  `json:"vocab_size"`
		BlankAsPad bool `json:"blank_as_pad"`
		Prednet    struct {
			Hidden int `json:"pred_hidden"`
			Layers int `json:"pred_rnn_layers"`
		} `json:"prednet"`
	} `json:"decoder"`
	Joint struct {
		Classes      int `json:"num_classes"`
		ExtraOutputs int `json:"num_extra_outputs"`
		Net          struct {
			Hidden        int    `json:"joint_hidden"`
			EncoderHidden int    `json:"encoder_hidden"`
			PredHidden    int    `json:"pred_hidden"`
			Activation    string `json:"activation"`
		} `json:"jointnet"`
	} `json:"joint"`
	Decoding struct {
		ModelType string `json:"model_type"`
		Durations []int  `json:"durations"`
		Greedy    struct {
			MaxSymbols int `json:"max_symbols"`
		} `json:"greedy"`
	} `json:"decoding"`
}

// LoadPhononModel loads the original model.fermion and its bundled config.json
// into the Parakeet FastConformer/TDT graph.
func LoadPhononModel(data, configData []byte, logw io.Writer) (ParakeetConfig, ParakeetWeights, error) {
	return loadPhononModel(data, configData, logw, false)
}

func loadPhononModel(data, configData []byte, logw io.Writer, keepPacked bool) (ParakeetConfig, ParakeetWeights, error) {
	meta, err := parsePhononConfig(configData)
	if err != nil {
		return ParakeetConfig{}, ParakeetWeights{}, err
	}
	cfg := meta.parakeetConfig()
	model, err := phonon.Parse(data)
	if err != nil {
		return cfg, ParakeetWeights{}, err
	}
	f32 := func(name string, shape ...int) ([]float32, error) {
		if name == "encoder.pos_enc.pe" {
			// The relative position table is a non-persistent buffer in
			// the official Transformers graph, so the file has no record.
			return phononPositionTable(cfg.Encoder.DModel, meta.Encoder.PositionLimit), nil
		}
		t, err := phononTensor(model, name, shape)
		if err != nil {
			return nil, err
		}
		return t.Decode()
	}
	f32Weight := func(name string, rows, cols int) (Weight, error) {
		t, err := phononTensor(model, name, []int{rows, cols})
		if err != nil {
			return Weight{}, err
		}
		if t.Kind == "five_value" {
			if keepPacked && strings.HasPrefix(name, "encoder.layers.") && t.NumElements() <= parakeetMaxProjectionElements {
				return Weight{Rows: rows, Cols: cols, phononProjection: &t}, nil
			}
			// Each value is zero or +/- one of the stored FP16 row scales.
			// The ordinary raw FP16 kernels can read these weights exactly.
			raw, err := t.DecodeF16()
			return Weight{Raw: raw, Rows: rows, Cols: cols, Type: GGMLTypeF16}, err
		}
		values, err := t.Decode()
		if err != nil {
			return Weight{}, err
		}
		return Weight{F32: values, Rows: rows, Cols: cols, Type: GGMLTypeF32}, nil
	}
	vocab := func() ([]string, error) { return phononTextPieces(meta.Labels), nil }
	return loadParakeetWeights(cfg, f32, f32Weight, vocab, logw)
}

func parsePhononConfig(data []byte) (phononConfig, error) {
	var meta phononConfig
	if len(data) > maxPhononConfigBytes {
		return meta, fmt.Errorf("phonon config: bundled JSON exceeds 1 MiB")
	}
	if err := json.Unmarshal(data, &meta); err != nil {
		return meta, fmt.Errorf("phonon config: %w", err)
	}
	if meta.ModelType != "parakeet" || meta.Fermion.ContainerFormat != phonon.Format || meta.Decoding.ModelType != "tdt" {
		return meta, fmt.Errorf("phonon config: expected a five-value Parakeet TDT checkpoint")
	}
	p, e, d, j := meta.Preprocessor, meta.Encoder, meta.Decoder, meta.Joint
	// Phonon 2.5 has the Parakeet TDT v3 geometry. Reject unknown frontends
	// and encoder layouts rather than constructing a different graph from
	// plausible dimensions or letting untrusted metadata size allocations.
	if p.SampleRate != 16000 || p.Features != 128 || p.NFFT != 512 || p.WindowSize != 0.025 ||
		p.WindowStride != 0.01 || p.Window != "hann" || p.Normalize != "per_feature" ||
		!p.Log || p.FrameSplicing != 1 || p.PadTo != 0 {
		return meta, fmt.Errorf("phonon config: unsupported log-mel frontend")
	}
	if e.Features != p.Features || e.Layers != 24 || e.Dim != 1024 || e.FFExpansion != 4 ||
		e.Heads != 8 || e.ConvKernel != 9 || e.Subsampling != "dw_striding" ||
		e.SubsamplingFactor != 8 || e.SubsamplingChannels != 256 || e.Attention != "rel_pos" ||
		!slices.Equal(e.AttentionContext, []int{-1, -1}) || e.PositionLimit != 5000 ||
		e.ConvNorm != "batch_norm" || e.UseBias || e.ScaleInput || e.CausalDownsampling {
		return meta, fmt.Errorf("phonon config: unsupported FastConformer encoder")
	}
	if d.VocabSize != 8192 || !d.BlankAsPad || d.Prednet.Hidden != 640 || d.Prednet.Layers != 2 ||
		j.Classes != d.VocabSize || j.Net.EncoderHidden != e.Dim || j.Net.PredHidden != d.Prednet.Hidden ||
		j.Net.Hidden != 640 || j.Net.Activation != "relu" ||
		!slices.Equal(meta.Decoding.Durations, []int{0, 1, 2, 3, 4}) ||
		j.ExtraOutputs != len(meta.Decoding.Durations) || meta.Decoding.Greedy.MaxSymbols != 10 {
		return meta, fmt.Errorf("phonon config: unsupported TDT decoder")
	}
	if len(meta.Labels) != d.VocabSize || slices.Contains(meta.Labels, "") {
		return meta, fmt.Errorf("phonon config: expected %d nonempty vocabulary pieces", d.VocabSize)
	}
	return meta, nil
}

// Keep token IDs aligned with the prediction embedding and joint head.
// Language and transcription controls have no printable text.
func phononTextPieces(labels []string) []string {
	for i, label := range labels {
		if label == "<unk>" || label == "<pad>" ||
			strings.HasPrefix(label, "<|") && strings.HasSuffix(label, "|>") {
			labels[i] = ""
		}
	}
	return labels
}

func (meta phononConfig) parakeetConfig() ParakeetConfig {
	p, e, d, j := meta.Preprocessor, meta.Encoder, meta.Decoder, meta.Joint
	return ParakeetConfig{
		Preprocessor: ParakeetPreprocessorConfig{
			SampleRate: p.SampleRate, NumMelBins: p.Features, NFFT: p.NFFT,
			WinLength:   int(math.Round(p.WindowSize * float64(p.SampleRate))),
			HopLength:   int(math.Round(p.WindowStride * float64(p.SampleRate))),
			PreEmphasis: 0.97, LogZeroGuard: 1.0 / (1 << 24),
		},
		Encoder: ParakeetEncoderConfig{
			DModel: e.Dim, DFF: e.Dim * e.FFExpansion, NHeads: e.Heads, HeadDim: e.Dim / e.Heads, NLayers: e.Layers,
			ConvKernelSize: e.ConvKernel, SubsamplingConvChans: e.SubsamplingChannels, SubsamplingFactor: e.SubsamplingFactor,
			Epsilon: 1e-5,
		},
		RNNT: ParakeetRNNTConfig{
			VocabSize: d.VocabSize, BlankID: d.VocabSize, PredHidden: d.Prednet.Hidden, PredEmbedDim: d.Prednet.Hidden,
			PredNumLayers: d.Prednet.Layers, JointDim: j.Net.Hidden, Durations: meta.Decoding.Durations,
			MaxSymbolsPerStep: meta.Decoding.Greedy.MaxSymbols,
		},
	}
}

// The Fermion artifact uses Transformers parameter names. The existing GGUF
// convention follows NeMo, which uses the same row-major tensor layout.
var phononNames = strings.NewReplacer(
	"encoder.pre_encode.conv.", "encoder.subsampling.layers.",
	"encoder.pre_encode.out.", "encoder.subsampling.linear.",
	".conv.batch_norm.", ".conv.norm.",
	".self_attn.linear_q.", ".self_attn.q_proj.",
	".self_attn.linear_k.", ".self_attn.k_proj.",
	".self_attn.linear_v.", ".self_attn.v_proj.",
	".self_attn.linear_out.", ".self_attn.o_proj.",
	".self_attn.linear_pos.", ".self_attn.relative_k_proj.",
	".self_attn.pos_bias_", ".self_attn.bias_",
	"decoder.prediction.embed.", "decoder.embedding.",
	"joint.pred.", "decoder.decoder_projector.",
	"joint.enc.", "encoder_projector.",
	"joint.joint_net.2.", "joint.head.",
)

func phononTensor(model *phonon.Model, name string, shape []int) (phonon.Tensor, error) {
	original := name
	if param, ok := strings.CutPrefix(name, "decoder.prediction.dec_rnn.lstm."); ok {
		suffix, kind, _ := strings.Cut(param, ".")
		name = "decoder.lstm." + kind + "_" + suffix
	} else {
		name = phononNames.Replace(name)
	}
	t, ok := model.Tensors[name]
	if !ok {
		return t, fmt.Errorf("missing Phonon tensor %q (Parakeet %q)", name, original)
	}
	if !slices.Equal(t.Shape, shape) {
		return t, fmt.Errorf("Phonon tensor %q has shape %v, expected %v", name, t.Shape, shape)
	}
	return t, nil
}

func phononPositionTable(dim, positions int) []float32 {
	center := positions - 1
	rows := 2*center + 1
	table := make([]float32, rows*dim)
	for channel := 0; channel < dim; channel += 2 {
		freq := 1 / float32(math.Pow(10000, float64(channel)/float64(dim)))
		for row := range rows {
			angle := float32(center-row) * freq
			s, c := math.Sincos(float64(angle))
			i := row*dim + channel
			table[i], table[i+1] = float32(s), float32(c)
		}
	}
	return table
}
