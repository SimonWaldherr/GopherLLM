package gopherllm

import "math"

// This file implements Parakeet's prediction network (a 2-layer LSTM),
// joint network, and TDT greedy decoding -- see parakeet.go's doc comment,
// points 4-5.
//
// LSTM gate order and equations follow PyTorch's nn.LSTM exactly (the
// checkpoint's weight_ih_l{N}/weight_hh_l{N} are PyTorch's own tensors,
// unconverted): gates stacked [input, forget, cell, output], each of width
// Hidden; i,f,o gates use sigmoid, the cell candidate g uses tanh.

type parakeetLSTMState struct {
	H, C [][]float32 // [layer][Hidden]
}

func newParakeetLSTMState(layers []ParakeetLSTMLayer, hidden int) *parakeetLSTMState {
	s := &parakeetLSTMState{H: make([][]float32, len(layers)), C: make([][]float32, len(layers))}
	for i := range layers {
		s.H[i] = make([]float32, hidden)
		s.C[i] = make([]float32, hidden)
	}
	return s
}

func (s *parakeetLSTMState) clone() *parakeetLSTMState {
	out := &parakeetLSTMState{H: make([][]float32, len(s.H)), C: make([][]float32, len(s.C))}
	for i := range s.H {
		out.H[i] = append([]float32(nil), s.H[i]...)
		out.C[i] = append([]float32(nil), s.C[i]...)
	}
	return out
}

func sigmoid(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }

// parakeetLSTMStep runs one input through every LSTM layer, updating state
// in place, and returns the top layer's hidden state (the prediction
// network's output for this step).
func parakeetLSTMStep(layers []ParakeetLSTMLayer, state *parakeetLSTMState, input []float32) []float32 {
	x := input
	for li, layer := range layers {
		hidden := len(state.H[li])
		var gatesI, gatesH []float32
		layer.WeightIH.MatvecInto(x, &gatesI)
		layer.WeightHH.MatvecInto(state.H[li], &gatesH)
		newH := make([]float32, hidden)
		newC := make([]float32, hidden)
		for j := 0; j < hidden; j++ {
			gi := gatesI[j] + gatesH[j] + layer.BiasIH[j] + layer.BiasHH[j]
			gf := gatesI[hidden+j] + gatesH[hidden+j] + layer.BiasIH[hidden+j] + layer.BiasHH[hidden+j]
			gg := gatesI[2*hidden+j] + gatesH[2*hidden+j] + layer.BiasIH[2*hidden+j] + layer.BiasHH[2*hidden+j]
			go_ := gatesI[3*hidden+j] + gatesH[3*hidden+j] + layer.BiasIH[3*hidden+j] + layer.BiasHH[3*hidden+j]
			i := sigmoid(gi)
			f := sigmoid(gf)
			g := float32(math.Tanh(float64(gg)))
			o := sigmoid(go_)
			c := f*state.C[li][j] + i*g
			newC[j] = c
			newH[j] = o * float32(math.Tanh(float64(c)))
		}
		state.H[li] = newH
		state.C[li] = newC
		x = newH
	}
	return x
}

// parakeetJoint combines one encoder frame and the prediction network's
// current output into TDT's joint logits: vocab_size+1 token logits
// (index VocabSize is the RNNT blank) followed by len(Durations) duration
// logits, matching joint.joint_net.2's output width (verified: 8193+5=8198
// for this checkpoint).
func parakeetJoint(cfg ParakeetRNNTConfig, jw ParakeetJointWeights, encFrame, predOut []float32) []float32 {
	var encProj, predProj []float32
	jw.Enc.MatvecInto(encFrame, &encProj)
	jw.Pred.MatvecInto(predOut, &predProj)
	combined := make([]float32, len(encProj))
	for i := range combined {
		v := encProj[i] + float32At(jw.EncBias, i) + predProj[i] + float32At(jw.PredBias, i)
		// NeMo's RNNTJoint activation is a configured choice (relu/tanh/
		// sigmoid, per its own docstring), read from the training YAML,
		// which this checkpoint's GGUF metadata doesn't carry -- "relu" is
		// what NeMo's published Conformer-Transducer/TDT example configs
		// use, not independently confirmed for this specific checkpoint.
		if v < 0 {
			v = 0
		}
		combined[i] = v
	}
	var out []float32
	jw.Out.MatvecInto(combined, &out)
	for i := range out {
		out[i] += float32At(jw.OutBias, i)
	}
	return out
}

func float32At(s []float32, i int) float32 {
	if i < len(s) {
		return s[i]
	}
	return 0
}

func argmaxF32(x []float32) (int, float32) {
	best, bestV := 0, x[0]
	for i, v := range x {
		if v > bestV {
			best, bestV = i, v
		}
	}
	return best, bestV
}

// ParakeetGreedyDecodeTDT runs TDT's greedy decoding loop over encoder
// output frames, returning the emitted (non-blank) token IDs in order.
// durationCursor advances by the predicted duration on every step (0 means
// "stay on this frame and try again" -- capped by MaxSymbolsPerStep to
// guarantee termination the way NeMo's own decoder bounds it); a positive
// duration or the blank symbol both consume at least one frame.
func ParakeetGreedyDecodeTDT(cfg ParakeetConfig, w ParakeetWeights, encoderFrames [][]float32) []int {
	state := newParakeetLSTMState(w.Decoder.LSTM, cfg.RNNT.PredHidden)
	// The prediction network's autoregressive input is the previously
	// emitted label; before any label has been emitted, NeMo feeds the
	// zero vector (there is no "start of sequence" embedding row).
	prevEmbed := make([]float32, cfg.RNNT.PredEmbedDim)
	predOut := parakeetLSTMStep(w.Decoder.LSTM, state, prevEmbed)

	var tokens []int
	t := 0
	symbolsThisFrame := 0
	for t < len(encoderFrames) {
		logits := parakeetJoint(cfg.RNNT, w.Joint, encoderFrames[t], predOut)
		tokenLogits := logits[:cfg.RNNT.VocabSize+1]
		durLogits := logits[cfg.RNNT.VocabSize+1:]
		tokenID, _ := argmaxF32(tokenLogits)
		durIdx, _ := argmaxF32(durLogits)
		duration := cfg.RNNT.Durations[durIdx]

		if tokenID != cfg.RNNT.BlankID {
			tokens = append(tokens, tokenID)
			var embedRow []float32
			w.Decoder.Embed.RowInto(tokenID, cfg.RNNT.PredEmbedDim, &embedRow)
			predOut = parakeetLSTMStep(w.Decoder.LSTM, state, embedRow)
		}

		symbolsThisFrame++
		if duration > 0 || tokenID == cfg.RNNT.BlankID || symbolsThisFrame >= cfg.RNNT.MaxSymbolsPerStep {
			if duration <= 0 {
				duration = 1 // blank or the per-frame symbol cap must still make progress
			}
			t += duration
			symbolsThisFrame = 0
		}
	}
	return tokens
}
