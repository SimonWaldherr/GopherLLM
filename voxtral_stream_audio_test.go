package gopherllm

import (
	"context"
	"math"
	"testing"
)

func TestVoxtralBatchedAttentionMatchesCausalReference(t *testing.T) {
	const n, past, heads, dim, window = 5, 7, 2, 8, 4
	const total = n + past
	q, out := make([][]float32, n), make([][]float32, n)
	for i := range q {
		q[i] = make([]float32, heads*dim)
		out[i] = make([]float32, heads*dim)
		for j := range q[i] {
			q[i][j] = float32(math.Sin(float64(i*17+j) * .7))
		}
	}
	k, v := make([]float32, heads*total*dim), make([]float32, heads*total*dim)
	for i := range k {
		k[i] = float32(math.Sin(float64(i) * .31))
		v[i] = float32(math.Cos(float64(i) * .71))
	}
	scale := float32(1 / math.Sqrt(dim))
	if !blasAttentionBatch(q, k, v, out, heads, dim, past, window, total, scale) {
		t.Skip("accelerated attention unavailable")
	}
	for row := range n {
		for h := range heads {
			want := make([]float64, dim)
			denom := 0.0
			for j := past + row - window + 1; j <= past+row; j++ {
				score := 0.0
				for d := range dim {
					score += float64(q[row][h*dim+d]) * float64(k[(h*total+j)*dim+d])
				}
				p := math.Exp(score * float64(scale))
				denom += p
				for d := range dim {
					want[d] += p * float64(v[(h*total+j)*dim+d])
				}
			}
			for d := range dim {
				if math.Abs(float64(out[row][h*dim+d])-want[d]/denom) > 1e-5 {
					t.Fatalf("row %d head %d dim %d differs", row, h, d)
				}
			}
		}
	}
}

func TestVoxtralStreamingEncoderMatchesOfflineAcrossBoundaries(t *testing.T) {
	cfg, w := buildTinyVoxtralRealtimeWeights()
	samples := make([]float32, 2117)
	for i := range samples {
		samples[i] = float32(.2 * math.Sin(float64(i)*.173))
	}
	raw := cfg.Mel.HopLength * 2 * cfg.Projector.DownsampleFactor
	want, err := EncodeAudioVoxtralRealtime(cfg, w, padVoxtralAudioOffline(samples, raw))
	if err != nil {
		t.Fatal(err)
	}
	for _, size := range []int{1, 7, 31, 128, 513, 4096} {
		s, err := newVoxtralStreamAudio(cfg)
		if err != nil {
			t.Fatal(err)
		}
		var got [][]float32
		for off := 0; off < len(samples); off += size {
			end := min(off+size, len(samples))
			rows, err := s.push(context.Background(), w, samples[off:end], false)
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, rows...)
			if len(s.samples) >= cfg.Mel.NFFT {
				t.Fatal("unbounded audio history")
			}
			if len(s.residual) >= cfg.Projector.DownsampleFactor {
				t.Fatal("unbounded adapter residual")
			}
			// The persistent per-head K/V cache's allocated capacity only
			// ever needs to cover window plus the single largest burst of
			// new encoder frames any one push has produced so far -- it
			// must not keep climbing with the total audio processed, which
			// is what the pre-cache implementation's raw sample count (up
			// to len(samples), two orders of magnitude bigger) would look
			// like if this regressed.
			if window := cfg.Encoder.SlidingWindow; window > 0 && s.encoder.stride > len(samples)/4 {
				t.Fatalf("unbounded encoder cache: stride=%d window=%d", s.encoder.stride, window)
			}
		}
		rows, err := s.push(context.Background(), w, nil, true)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, rows...)
		if len(got) != len(want) {
			t.Fatalf("chunk %d: frames %d want %d", size, len(got), len(want))
		}
		for i := range want {
			for j, v := range want[i] {
				if math.Abs(float64(got[i][j]-v)) > 2e-5 {
					t.Fatalf("chunk %d frame %d dim %d got %g want %g", size, i, j, got[i][j], v)
				}
			}
		}
	}
}

func TestVoxtralStreamingDecoderKeepsStateAndFlushes(t *testing.T) {
	cfg, w := buildTinyVoxtralRealtimeWeights()
	cfg.StreamingPadTokenID = 3
	vocab := make([]string, cfg.Decoder.VocabSize)
	for i := range vocab {
		vocab[i] = "a"
	}
	makeSession := func() *VoxtralRealtimeSession {
		return &VoxtralRealtimeSession{cfg: cfg, weights: w, tokenizer: &Tokenizer{Vocab: vocab}, bosID: 1, eosID: 2}
	}
	pcm := make([]float32, 1927)
	for i := range pcm {
		pcm[i] = float32(math.Sin(float64(i) * .1))
	}
	whole := makeSession()
	want, err := whole.Push(context.Background(), pcm, true)
	if err != nil {
		t.Fatal(err)
	}
	chunks := makeSession()
	var state *VoxtralRealtimeDecoderState
	var got string
	for off := 0; off < len(pcm); off += 37 {
		got, err = chunks.Push(context.Background(), pcm[off:min(off+37, len(pcm))], false)
		if err != nil {
			t.Fatal(err)
		}
		if state != nil && state != chunks.decoder {
			t.Fatal("decoder recreated across chunks")
		}
		state = chunks.decoder
	}
	got, err = chunks.Push(context.Background(), nil, true)
	if err != nil {
		t.Fatal(err)
	}
	if got != want || chunks.pos != whole.pos {
		t.Fatalf("chunked transcript/positions differ: %q %q, %d %d", got, want, chunks.pos, whole.pos)
	}
	if _, err = chunks.Push(context.Background(), pcm, false); err == nil {
		t.Fatal("accepted audio after finish")
	}
}
