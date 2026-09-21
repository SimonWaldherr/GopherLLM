package gopherllm

import (
	"fmt"
	"io"
)

// LoadParakeetModel loads a Parakeet-TDT GGUF's config and weights -- see
// parakeet.go's doc comment for the architecture and tensor-family
// overview. Every 2D linear weight loads as F32 the same way voxtral's
// loaders already do (loadF32Vec/loadWeight's F32 branch is dtype-agnostic
// up to the quantized-kernel dispatch this doesn't need, since
// Weight.MatvecInto infers rows/cols from the caller's input length for
// F32 weights regardless of quantization). No axis permutation is needed
// for any tensor here: every multi-dimensional tensor inspected (the
// subsampling Conv2d weights, the per-head pos_bias_u/v vectors, the
// per-channel depthwise_conv kernel) has GGUF-native dims already ordered
// so that a plain sequential byte copy reproduces PyTorch's own row-major
// layout -- verified per-tensor against the shapes reported by
// --list-tensors, the same way voxtral_official.go's safetensors path
// needed none either, unlike GGUF's "voxtral.*" convention's transposed
// mel filterbank.
func LoadParakeetModel(data []byte, gguf *GGUFFile, logw io.Writer) (ParakeetConfig, ParakeetWeights, error) {
	if logw == nil {
		logw = io.Discard
	}
	config, err := loadParakeetConfig(gguf)
	if err != nil {
		return ParakeetConfig{}, ParakeetWeights{}, err
	}

	tensors := indexTensors(gguf)
	inferred := inferTensorSizes(data, gguf)
	f32 := func(name string) ([]float32, error) {
		return loadF32Vec(data, gguf.DataOffset, name, tensors, inferred)
	}
	f32Weight := func(name string) (Weight, error) {
		w, err := loadWeight(data, gguf.DataOffset, name, tensors, inferred, true, false, false, false)
		if err != nil {
			return Weight{}, err
		}
		if w.F32 == nil {
			return Weight{}, fmt.Errorf("expected f32-decodable tensor for %s", name)
		}
		return w, nil
	}

	weights := ParakeetWeights{}
	weights.MelFilterbank = voxtralSlaneyMelFilterbank(config.Preprocessor.SampleRate, config.Preprocessor.NFFT, config.Preprocessor.NumMelBins, 0, float64(config.Preprocessor.SampleRate)/2)

	// Subsampling stem: conv.0 is a regular Conv2d (in_per_group ==
	// in_channels == 1, the raw mel "image"); conv.2/conv.5 are depthwise
	// (in_per_group == 1, one filter per channel); conv.3/conv.6 are
	// pointwise 1x1 convs that mix the depthwise output back across
	// channels (in_per_group == out_channels). Index gaps (1, 4, 7) are the
	// checkpoint's own nn.Sequential slots for the (parameter-free) ReLU
	// activations between stages.
	chans := config.Encoder.SubsamplingConvChans
	convSpecs := []struct {
		idx                    int
		outCh, inPerGroup      int
		kh, kw, sh, sw, ph, pw int
	}{
		{0, chans, 1, 3, 3, 2, 2, 1, 1},
		{2, chans, 1, 3, 3, 2, 2, 1, 1},
		{3, chans, chans, 1, 1, 1, 1, 0, 0},
		{5, chans, 1, 3, 3, 2, 2, 1, 1},
		{6, chans, chans, 1, 1, 1, 1, 0, 0},
	}
	for i, spec := range convSpecs {
		wName := fmt.Sprintf("encoder.pre_encode.conv.%d.weight", spec.idx)
		bName := fmt.Sprintf("encoder.pre_encode.conv.%d.bias", spec.idx)
		w, err := f32(wName)
		if err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if want := spec.outCh * spec.inPerGroup * spec.kh * spec.kw; len(w) != want {
			return config, weights, fmt.Errorf("loading parakeet model: tensor %s has %d elements, want %d (out=%d in_per_group=%d kh=%d kw=%d)", wName, len(w), want, spec.outCh, spec.inPerGroup, spec.kh, spec.kw)
		}
		b, err := f32(bName)
		if err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		weights.Encoder.SubsamplingConvs[i] = ParakeetSubsamplingConv{
			Weight: w, Bias: b, OutChannels: spec.outCh, InPerGroup: spec.inPerGroup,
			KH: spec.kh, KW: spec.kw, StrideH: spec.sh, StrideW: spec.sw, PadH: spec.ph, PadW: spec.pw,
		}
	}
	if weights.Encoder.SubsamplingOut, err = f32Weight("encoder.pre_encode.out.weight"); err != nil {
		return config, weights, fmt.Errorf("loading parakeet model: %w", err)
	}
	if weights.Encoder.SubsamplingOutBias, err = f32("encoder.pre_encode.out.bias"); err != nil {
		return config, weights, fmt.Errorf("loading parakeet model: %w", err)
	}
	if weights.Encoder.PosEnc, err = f32("encoder.pos_enc.pe"); err != nil {
		return config, weights, fmt.Errorf("loading parakeet model: %w", err)
	}
	if config.Encoder.DModel > 0 {
		weights.Encoder.PosEncMaxLen = len(weights.Encoder.PosEnc) / config.Encoder.DModel
	}

	layers := make([]ParakeetEncoderLayer, 0, config.Encoder.NLayers)
	for l := range config.Encoder.NLayers {
		p := fmt.Sprintf("encoder.layers.%d.", l)
		var layer ParakeetEncoderLayer
		if layer.NormFeedForward1Weight, err = f32(p + "norm_feed_forward1.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.NormFeedForward1Bias, err = f32(p + "norm_feed_forward1.bias"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.FeedForward1Linear1, err = f32Weight(p + "feed_forward1.linear1.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.FeedForward1Linear2, err = f32Weight(p + "feed_forward1.linear2.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.NormConvWeight, err = f32(p + "norm_conv.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.NormConvBias, err = f32(p + "norm_conv.bias"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.Conv.PointwiseConv1, err = f32Weight(p + "conv.pointwise_conv1.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.Conv.DepthwiseConv, err = f32(p + "conv.depthwise_conv.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.Conv.BatchNormWeight, err = f32(p + "conv.batch_norm.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.Conv.BatchNormBias, err = f32(p + "conv.batch_norm.bias"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.Conv.BatchNormMean, err = f32(p + "conv.batch_norm.running_mean"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.Conv.BatchNormVar, err = f32(p + "conv.batch_norm.running_var"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.Conv.PointwiseConv2, err = f32Weight(p + "conv.pointwise_conv2.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.NormSelfAttWeight, err = f32(p + "norm_self_att.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.NormSelfAttBias, err = f32(p + "norm_self_att.bias"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.SelfAttn.LinearQ, err = f32Weight(p + "self_attn.linear_q.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.SelfAttn.LinearK, err = f32Weight(p + "self_attn.linear_k.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.SelfAttn.LinearV, err = f32Weight(p + "self_attn.linear_v.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.SelfAttn.LinearOut, err = f32Weight(p + "self_attn.linear_out.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.SelfAttn.LinearPos, err = f32Weight(p + "self_attn.linear_pos.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.SelfAttn.PosBiasU, err = f32(p + "self_attn.pos_bias_u"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.SelfAttn.PosBiasV, err = f32(p + "self_attn.pos_bias_v"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.NormFeedForward2Weight, err = f32(p + "norm_feed_forward2.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.NormFeedForward2Bias, err = f32(p + "norm_feed_forward2.bias"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.FeedForward2Linear1, err = f32Weight(p + "feed_forward2.linear1.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.FeedForward2Linear2, err = f32Weight(p + "feed_forward2.linear2.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.NormOutWeight, err = f32(p + "norm_out.weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.NormOutBias, err = f32(p + "norm_out.bias"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		layers = append(layers, layer)
		if l == 0 || l+1 == config.Encoder.NLayers || (l+1)%8 == 0 {
			fmt.Fprintf(logw, "  Loaded parakeet encoder layer %d/%d\n", l+1, config.Encoder.NLayers)
		}
	}
	weights.Encoder.Layers = layers

	if weights.Decoder.Embed, err = f32Weight("decoder.prediction.embed.weight"); err != nil {
		return config, weights, fmt.Errorf("loading parakeet model: %w", err)
	}
	decLayers := make([]ParakeetLSTMLayer, 0, config.RNNT.PredNumLayers)
	for l := range config.RNNT.PredNumLayers {
		const p = "decoder.prediction.dec_rnn.lstm."
		suffix := fmt.Sprintf("l%d", l)
		var layer ParakeetLSTMLayer
		if layer.WeightIH, err = f32Weight(p + "ih_" + suffix + ".weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.WeightHH, err = f32Weight(p + "hh_" + suffix + ".weight"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.BiasIH, err = f32(p + "ih_" + suffix + ".bias"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		if layer.BiasHH, err = f32(p + "hh_" + suffix + ".bias"); err != nil {
			return config, weights, fmt.Errorf("loading parakeet model: %w", err)
		}
		decLayers = append(decLayers, layer)
	}
	weights.Decoder.LSTM = decLayers

	if weights.Joint.Pred, err = f32Weight("joint.pred.weight"); err != nil {
		return config, weights, fmt.Errorf("loading parakeet model: %w", err)
	}
	if weights.Joint.PredBias, err = f32("joint.pred.bias"); err != nil {
		return config, weights, fmt.Errorf("loading parakeet model: %w", err)
	}
	if weights.Joint.Enc, err = f32Weight("joint.enc.weight"); err != nil {
		return config, weights, fmt.Errorf("loading parakeet model: %w", err)
	}
	if weights.Joint.EncBias, err = f32("joint.enc.bias"); err != nil {
		return config, weights, fmt.Errorf("loading parakeet model: %w", err)
	}
	if weights.Joint.Out, err = f32Weight("joint.joint_net.2.weight"); err != nil {
		return config, weights, fmt.Errorf("loading parakeet model: %w", err)
	}
	if weights.Joint.OutBias, err = f32("joint.joint_net.2.bias"); err != nil {
		return config, weights, fmt.Errorf("loading parakeet model: %w", err)
	}

	vocabValue, ok := gguf.Metadata["asr.tokenizer.vocab"]
	if !ok {
		return config, weights, fmt.Errorf("loading parakeet model: missing asr.tokenizer.vocab")
	}
	weights.Vocab, ok = vocabValue.AsStringArray()
	if !ok {
		return config, weights, fmt.Errorf("loading parakeet model: asr.tokenizer.vocab is not a string array")
	}
	if len(weights.Vocab) != config.RNNT.VocabSize {
		return config, weights, fmt.Errorf("loading parakeet model: asr.tokenizer.vocab has %d entries, want VocabSize=%d", len(weights.Vocab), config.RNNT.VocabSize)
	}

	return config, weights, nil
}
