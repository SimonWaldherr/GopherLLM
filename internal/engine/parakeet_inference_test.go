package engine

import (
	"math"
	"reflect"
	"testing"
)

func referenceParakeetLSTMStep(layers []ParakeetLSTMLayer, h, c [][]float32, input []float32) []float32 {
	x := input
	for l, layer := range layers {
		n := len(h[l])
		var gi, gh []float32
		layer.WeightIH.MatvecInto(x, &gi)
		layer.WeightHH.MatvecInto(h[l], &gh)
		nh, nc := make([]float32, n), make([]float32, n)
		for j := 0; j < n; j++ {
			i := sigmoid(gi[j] + gh[j] + layer.BiasIH[j] + layer.BiasHH[j])
			f := sigmoid(gi[n+j] + gh[n+j] + layer.BiasIH[n+j] + layer.BiasHH[n+j])
			g := float32(math.Tanh(float64(gi[2*n+j] + gh[2*n+j] + layer.BiasIH[2*n+j] + layer.BiasHH[2*n+j])))
			o := sigmoid(gi[3*n+j] + gh[3*n+j] + layer.BiasIH[3*n+j] + layer.BiasHH[3*n+j])
			nc[j] = f*c[l][j] + i*g
			nh[j] = o * float32(math.Tanh(float64(nc[j])))
		}
		h[l], c[l], x = nh, nc, nh
	}
	return x
}

func TestParakeetLSTMReusedStateMatchesReference(t *testing.T) {
	cfg, w, _ := buildParakeetDecodeFixture(8, 1, false)
	state := newParakeetLSTMState(w.Decoder.LSTM, cfg.RNNT.PredHidden)
	h, c := make([][]float32, 2), make([][]float32, 2)
	for l := range h {
		h[l], c[l] = make([]float32, 8), make([]float32, 8)
	}
	for step := 0; step < 12; step++ {
		input := make([]float32, 8)
		for i := range input {
			input[i] = float32(math.Sin(float64(step*7 + i*13)))
		}
		want := referenceParakeetLSTMStep(w.Decoder.LSTM, h, c, input)
		got := parakeetLSTMStep(w.Decoder.LSTM, state, input)
		if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(state.C, c) {
			t.Fatalf("step %d: recurrent state differs", step)
		}
	}
}

func TestParakeetProjectedJointMatchesReference(t *testing.T) {
	cfg, w, frames := buildParakeetDecodeFixture(8, 4, false)
	var scratch parakeetJointScratch
	for i := range frames {
		pred := frames[len(frames)-1-i]
		want := parakeetJoint(cfg.RNNT, w.Joint, frames[i], pred)
		w.Joint.Enc.MatvecInto(frames[i], &scratch.enc)
		w.Joint.Pred.MatvecInto(pred, &scratch.pred)
		got := parakeetJointProjectedInto(w.Joint, &scratch)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("frame %d: joint logits differ", i)
		}
	}
}

func TestParakeetFeedForwardMatchesPerFrameReference(t *testing.T) {
	const dim, hidden = 4, 7
	fill := func(n int) []float32 {
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(i%11-5) / 20
		}
		return data
	}
	w1, w2 := Weight{F32: fill(hidden * dim)}, Weight{F32: fill(dim * hidden)}
	normW, normB := []float32{1, .8, 1.2, -.6}, []float32{.1, -.2, .3, -.4}
	xs := [][]float32{{.2, -.7, 1.1, -.3}, {-.4, .3, -.8, .9}, {1.2, -.1, .6, -.7}}
	before := make([][]float32, len(xs))
	for i := range xs {
		before[i] = append([]float32(nil), xs[i]...)
	}
	got := parakeetFeedForwardBatch(xs, normW, normB, w1, w2, 1e-5)
	for i := range xs {
		var norm, act, proj []float32
		layerNormInto(xs[i], normW, normB, 1e-5, &norm)
		w1.MatvecInto(norm, &act)
		for j, v := range act {
			act[j] = siluExact(v)
		}
		w2.MatvecInto(act, &proj)
		for j, v := range proj {
			want := xs[i][j] + .5*v
			if math.Abs(float64(got[i][j]-want)) > 1e-6 {
				t.Fatalf("[%d][%d]: got %g, want %g", i, j, got[i][j], want)
			}
		}
	}
	if !reflect.DeepEqual(xs, before) {
		t.Fatal("feed-forward modified its input")
	}
}

func TestParakeetConvModuleMatchesPerFrameReference(t *testing.T) {
	const dim, kernel = 4, 3
	fill := func(n int) []float32 {
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(i%11-5) / 20
		}
		return data
	}
	conv := ParakeetEncoderConvLayer{PointwiseConv1: Weight{F32: fill(2 * dim * dim)}, PointwiseConv2: Weight{F32: fill(dim * dim)}, DepthwiseConv: fill(dim * kernel), BatchNormMean: []float32{.1, -.2, .3, -.4}, BatchNormVar: []float32{.8, 1.2, .7, 1.1}, BatchNormWeight: []float32{1, .9, -.8, 1.3}, BatchNormBias: fill(dim)}
	normW, normB := []float32{1, .8, 1.2, -.6}, []float32{.1, -.2, .3, -.4}
	xs := [][]float32{{.2, -.7, 1.1, -.3}, {-.4, .3, -.8, .9}, {1.2, -.1, .6, -.7}}
	glu := make([][]float32, len(xs))
	for i := range xs {
		var norm, expanded []float32
		layerNormInto(xs[i], normW, normB, 1e-5, &norm)
		conv.PointwiseConv1.MatvecInto(norm, &expanded)
		glu[i] = make([]float32, dim)
		for c := 0; c < dim; c++ {
			glu[i][c] = expanded[c] * sigmoid(expanded[dim+c])
		}
	}
	got := parakeetConvModule(xs, conv, normW, normB, dim, kernel, 1e-5)
	if batchSlicesOverlap(got[0], xs[0]) {
		t.Fatal("convolution result aliases its input")
	}
	for i := range xs {
		depth := make([]float32, dim)
		for c := range depth {
			for k := 0; k < kernel; k++ {
				source := i - kernel/2 + k
				if source >= 0 && source < len(xs) {
					depth[c] += glu[source][c] * conv.DepthwiseConv[c*kernel+k]
				}
			}
			norm := (depth[c] - conv.BatchNormMean[c]) / float32(math.Sqrt(float64(conv.BatchNormVar[c])+1e-5))
			depth[c] = siluExact(norm*conv.BatchNormWeight[c] + conv.BatchNormBias[c])
		}
		var proj []float32
		conv.PointwiseConv2.MatvecInto(depth, &proj)
		for c, v := range proj {
			want := xs[i][c] + v
			if math.Abs(float64(got[i][c]-want)) > 1e-6 {
				t.Fatalf("[%d][%d]: got %g, want %g", i, c, got[i][c], want)
			}
		}
	}
}

func TestParakeetRelativePositionScoresUseEachHead(t *testing.T) {
	const heads, headDim, dim, count, posLen = 2, 2, 4, 3, 7
	identity := make([]float32, dim*dim)
	for i := 0; i < dim; i++ {
		identity[i*dim+i] = 1
	}
	linear := Weight{F32: identity}
	attn := ParakeetEncoderAttention{LinearQ: linear, LinearK: linear, LinearV: linear, LinearPos: linear, LinearOut: linear, PosBiasU: []float32{.1, .2, .3, .4}, PosBiasV: []float32{-.2, .4, -.6, .8}}
	xs := [][]float32{{.2, -.7, 1.1, -.3}, {-.4, .3, -.8, .9}, {1.2, -.1, .6, -.7}}
	pos := make([]float32, posLen*dim)
	for i := range pos {
		pos[i] = float32((i*7)%31-15) / 10
	}
	got, err := parakeetRelPosSelfAttention(xs, attn, []float32{1, 1, 1, 1}, nil, heads, headDim, pos, posLen, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	if batchSlicesOverlap(got[0], xs[0]) {
		t.Fatal("attention result aliases its input")
	}
	normed := make([][]float32, count)
	for i := range xs {
		layerNormInto(xs[i], []float32{1, 1, 1, 1}, nil, 1e-5, &normed[i])
	}
	for qi := 0; qi < count; qi++ {
		for head := 0; head < heads; head++ {
			off := head * headDim
			scores := make([]float32, count)
			largest := negMaxF32
			for ki := 0; ki < count; ki++ {
				// Relative index is center-qi+ki in the full position
				// table; every head selects its own channel interval.
				ac, bd := float32(0), float32(0)
				for d := 0; d < headDim; d++ {
					ac += (normed[qi][off+d] + attn.PosBiasU[off+d]) * normed[ki][off+d]
					bd += (normed[qi][off+d] + attn.PosBiasV[off+d]) * pos[(posLen/2-qi+ki)*dim+off+d]
				}
				scores[ki] = (ac + bd) * float32(1/math.Sqrt(headDim))
				largest = max(largest, scores[ki])
			}
			denom := float32(0)
			for ki := range scores {
				scores[ki] = fastExpF32(scores[ki] - largest)
				denom += scores[ki]
			}
			for d := 0; d < headDim; d++ {
				want := xs[qi][off+d]
				for ki := 0; ki < count; ki++ {
					want += scores[ki] * (1 / denom) * normed[ki][off+d]
				}
				if math.Abs(float64(got[qi][off+d]-want)) > 2e-6 {
					t.Fatalf("query=%d head=%d channel=%d: got %g, want %g", qi, head, d, got[qi][off+d], want)
				}
			}
		}
	}
}

func buildParakeetDecodeFixture(hidden, frames int, blank bool) (ParakeetConfig, ParakeetWeights, [][]float32) {
	const vocab = 16
	cfg := ParakeetConfig{RNNT: ParakeetRNNTConfig{VocabSize: vocab, BlankID: vocab, PredHidden: hidden, PredEmbedDim: hidden, PredNumLayers: 2, JointDim: hidden, Durations: []int{0, 1, 2}, MaxSymbolsPerStep: 3}}
	fill := func(rows, cols, seed int) Weight {
		data := make([]float32, rows*cols)
		for i := range data {
			data[i] = float32((i*7+seed)%19-9) / float32(cols*3)
		}
		return Weight{F32: data}
	}
	w := ParakeetWeights{Decoder: ParakeetDecoderWeights{Embed: fill(vocab+1, hidden, 1)}, Joint: ParakeetJointWeights{Enc: fill(hidden, hidden, 2), Pred: fill(hidden, hidden, 3), Out: fill(vocab+1+len(cfg.RNNT.Durations), hidden, 4), EncBias: make([]float32, hidden), PredBias: make([]float32, hidden), OutBias: make([]float32, vocab+1+len(cfg.RNNT.Durations))}}
	for l := 0; l < 2; l++ {
		w.Decoder.LSTM = append(w.Decoder.LSTM, ParakeetLSTMLayer{WeightIH: fill(4*hidden, hidden, 5+l), WeightHH: fill(4*hidden, hidden, 7+l), BiasIH: make([]float32, 4*hidden), BiasHH: make([]float32, 4*hidden)})
	}
	// Exercise both blank spans (prediction state stays fixed) and zero
	// durations (several symbols reuse the same encoder frame).
	if blank {
		w.Joint.OutBias[vocab] = 1
		w.Joint.OutBias[vocab+2] = 1
	} else {
		w.Joint.OutBias[3] = 1
		w.Joint.OutBias[vocab+1] = 1
	}
	xs := make([][]float32, frames)
	for i := range xs {
		xs[i] = make([]float32, hidden)
		for j := range xs[i] {
			xs[i][j] = float32((i*17+j*3)%23-11) / 11
		}
	}
	return cfg, w, xs
}

// This reference keeps the original per-step projection behavior so cached
// projections must produce exactly the same greedy decisions.
func referenceParakeetGreedyDecode(cfg ParakeetConfig, w ParakeetWeights, frames [][]float32) []int {
	state := newParakeetLSTMState(w.Decoder.LSTM, cfg.RNNT.PredHidden)
	pred := parakeetLSTMStep(w.Decoder.LSTM, state, make([]float32, cfg.RNNT.PredEmbedDim))
	var tokens []int
	t, symbols := 0, 0
	for t < len(frames) {
		logits := parakeetJoint(cfg.RNNT, w.Joint, frames[t], pred)
		token, _ := argmaxF32(logits[:cfg.RNNT.VocabSize+1])
		dur, _ := argmaxF32(logits[cfg.RNNT.VocabSize+1:])
		duration := cfg.RNNT.Durations[dur]
		if token != cfg.RNNT.BlankID {
			tokens = append(tokens, token)
			var embed []float32
			w.Decoder.Embed.RowInto(token, cfg.RNNT.PredEmbedDim, &embed)
			pred = parakeetLSTMStep(w.Decoder.LSTM, state, embed)
		}
		symbols++
		if duration > 0 || token == cfg.RNNT.BlankID || symbols >= cfg.RNNT.MaxSymbolsPerStep {
			if duration <= 0 {
				duration = 1
			}
			t += duration
			symbols = 0
		}
	}
	return tokens
}

func TestParakeetGreedyDecodeProjectionReuse(t *testing.T) {
	for _, blank := range []bool{false, true} {
		cfg, w, frames := buildParakeetDecodeFixture(8, 7, blank)
		want := referenceParakeetGreedyDecode(cfg, w, frames)
		got := ParakeetGreedyDecodeTDT(cfg, w, frames)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("blank=%v: got %v, want %v", blank, got, want)
		}
		if !blank && len(got) != len(frames)*cfg.RNNT.MaxSymbolsPerStep {
			t.Fatalf("zero-duration fixture emitted %d tokens", len(got))
		}
	}
}

func BenchmarkParakeetDecodeProjectionReuse(b *testing.B) {
	for _, blank := range []bool{false, true} {
		name := "Symbols"
		if blank {
			name = "BlankSpan"
		}
		b.Run(name, func(b *testing.B) {
			cfg, w, frames := buildParakeetDecodeFixture(128, 44, blank)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ParakeetGreedyDecodeTDT(cfg, w, frames)
			}
		})
	}
}
