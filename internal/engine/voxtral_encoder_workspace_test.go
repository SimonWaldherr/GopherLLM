package engine

import (
	"context"
	"errors"
	"math"
	"os"
	"strconv"
	"testing"
)

func TestVoxtralFFNGateReuse(t *testing.T) {
	for _, n := range []int{0, 1, 31, 32, 33, 5120} {
		gate, up := randomExpertInput(n, 83701), randomExpertInput(n, 83702)
		want := make([]float32, n)
		siluMulF32(gate, up, want)
		storage := make([]float32, n+2)
		storage[0], storage[n+1] = 123, 456
		got := storage[1 : n+1]
		copy(got, gate)
		siluMulF32(got, up, got)
		for i, value := range want {
			if math.Float32bits(value) != math.Float32bits(got[i]) {
				t.Fatalf("width%d element%d: reused gate changed activation", n, i)
			}
		}
		if storage[0] != 123 || storage[n+1] != 456 {
			t.Fatal("activation overwrote a guard")
		}
	}
}

func voxtralEncoderTestRows(start, n, dim int) [][]float32 {
	rows := audioMatrixRows(n, dim)
	for i := range rows {
		for j := range rows[i] {
			rows[i][j] = float32(math.Sin(float64((start+i)*dim+j)*.173)) * .2
		}
	}
	return rows
}

func voxtralEncoderTestChannels(rows [][]float32, dim int) []float32 {
	h := make([]float32, len(rows)*dim)
	for t, row := range rows {
		for c := range dim {
			h[c*len(rows)+t] = row[c]
		}
	}
	return h
}

func checkVoxtralEncoderWorkspace(t *testing.T, cfg VoxtralRealtimeConfig, w VoxtralRealtimeWeights, sizes []int) {
	t.Helper()
	ctx := context.Background()
	var state, fresh voxtralEncoderState
	var input, retained, snapshots [][]float32
	for _, n := range sizes {
		start := len(input)
		rows := voxtralEncoderTestRows(start, n, cfg.Encoder.DModel)
		for _, row := range rows {
			input = append(input, append([]float32(nil), row...))
		}
		// Reconstructing the channel-major input and discarding all scratch
		// each time reproduces the previous encoder handoff and workspace.
		fresh.scratch = voxtralEncoderScratch{}
		want, err := forwardVoxtralEncoderChunk(ctx, cfg, w, voxtralEncoderTestChannels(rows, cfg.Encoder.DModel), n, &fresh)
		if err != nil {
			t.Fatal(err)
		}
		got, err := forwardVoxtralEncoderRowsChunk(ctx, cfg, w, rows, &state)
		if err != nil {
			t.Fatal(err)
		}
		// A complete offline prefix also checks absolute RoPE positions and
		// eviction independently of either incremental state's workspace.
		offline, err := forwardVoxtralEncoderChunk(ctx, cfg, w, voxtralEncoderTestChannels(input, cfg.Encoder.DModel), len(input), nil)
		if err != nil {
			t.Fatal(err)
		}
		for i, row := range got {
			for j, v := range row {
				if math.Float32bits(v) != math.Float32bits(want[i][j]) {
					t.Fatalf("chunk %d row %d dim %d: reused %g, fresh %g", start, i, j, v, want[i][j])
				}
				if d := math.Abs(float64(v - offline[start+i][j])); d > 3e-5 || math.IsNaN(d) {
					t.Fatalf("chunk %d row %d dim %d: incremental %g, offline %g", start, i, j, v, offline[start+i][j])
				}
			}
			retained = append(retained, row)
			snapshots = append(snapshots, append([]float32(nil), row...))
		}
		for i, row := range retained {
			for j, v := range row {
				if math.Float32bits(v) != math.Float32bits(snapshots[i][j]) {
					t.Fatalf("later push overwrote retained encoder row %d dim %d", i, j)
				}
			}
		}
		if state.pos != len(input) || fresh.pos != state.pos {
			t.Fatalf("positions: reused=%d fresh=%d input=%d", state.pos, fresh.pos, len(input))
		}
	}
}

func TestVoxtralEncoderWorkspaceAcrossChunkSizes(t *testing.T) {
	for _, interleaved := range []bool{false, true} {
		name := "split"
		if interleaved {
			name = "interleaved"
		}
		t.Run(name, func(t *testing.T) {
			cfg, w := buildTinyVoxtralRealtimeWeights()
			cfg.RopeInterleaved = interleaved
			checkVoxtralEncoderWorkspace(t, cfg, w, []int{2, 7, 1, 4, 9, 2, 7})
		})
	}
}

func TestVoxtralEncoderWorkspaceCanceledBeforeMutation(t *testing.T) {
	cfg, w := buildTinyVoxtralRealtimeWeights()
	var state voxtralEncoderState
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	rows := voxtralEncoderTestRows(0, 3, cfg.Encoder.DModel)
	want := voxtralEncoderTestRows(0, 3, cfg.Encoder.DModel)
	if _, err := forwardVoxtralEncoderRowsChunk(ctx, cfg, w, rows, &state); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	if state.pos != 0 || state.kHead != nil || state.scratch.q.data != nil {
		t.Fatal("canceled call mutated encoder state")
	}
	for i := range rows {
		for j, v := range rows[i] {
			if v != want[i][j] {
				t.Fatal("canceled call mutated input")
			}
		}
	}
	if _, err := forwardVoxtralEncoderRowsChunk(context.Background(), cfg, w, rows, &state); err != nil {
		t.Fatal(err)
	}
	kBefore := append([]float32(nil), state.kHead[0]...)
	qBefore := append([]float32(nil), state.scratch.q.data...)
	if _, err := forwardVoxtralEncoderRowsChunk(ctx, cfg, w, want, &state); !errors.Is(err, context.Canceled) {
		t.Fatalf("warm state cancellation: %v", err)
	}
	if state.pos != len(rows) {
		t.Fatal("canceled call advanced the existing cache")
	}
	for i, v := range kBefore {
		if state.kHead[0][i] != v {
			t.Fatal("canceled call changed the existing cache")
		}
	}
	for i, v := range qBefore {
		if state.scratch.q.data[i] != v {
			t.Fatal("canceled call changed the existing scratch")
		}
	}
}

func TestVoxtralEncoderWorkspacesAreIndependent(t *testing.T) {
	cfg, w := buildTinyVoxtralRealtimeWeights()
	var first, second voxtralEncoderState
	ctx := context.Background()
	if _, err := forwardVoxtralEncoderRowsChunk(ctx, cfg, w, voxtralEncoderTestRows(0, 3, cfg.Encoder.DModel), &first); err != nil {
		t.Fatal(err)
	}
	q := append([]float32(nil), first.scratch.q.data...)
	if _, err := forwardVoxtralEncoderRowsChunk(ctx, cfg, w, voxtralEncoderTestRows(100, 3, cfg.Encoder.DModel), &second); err != nil {
		t.Fatal(err)
	}
	for i, v := range q {
		if first.scratch.q.data[i] != v {
			t.Fatal("second stream overwrote first stream's scratch")
		}
	}
	if &first.scratch.q.data[0] == &second.scratch.q.data[0] {
		t.Fatal("streams share scratch")
	}
}

func TestVoxtralEncoderWorkspaceRetentionBound(t *testing.T) {
	var scratch voxtralEncoderScratch
	scratch.q.resize(100, 8)
	scratch.ropeSin = make([]float32, 100)
	size := scratch.retainedBytes()
	// Shrinking visible lengths must not hide retained backing storage.
	scratch.q.resize(1, 8)
	scratch.ropeSin = scratch.ropeSin[:1]
	if scratch.retainedBytes() != size {
		t.Fatal("retention accounting ignored capacity")
	}
	scratch.releaseIfOversized(size)
	if scratch.q.data == nil {
		t.Fatal("workspace at the limit was discarded")
	}
	scratch.releaseIfOversized(size - 1)
	if scratch.retainedBytes() != 0 || scratch.q.data != nil {
		t.Fatal("oversized workspace retained its backing arrays")
	}
}

func TestVoxtralAudioMatrixResizeDropsReplacedStorage(t *testing.T) {
	var matrix voxtralAudioMatrix
	matrix.resize(4, 4)
	old := &matrix.data[0]
	rows := matrix.resize(1, 32)
	if len(rows) != 1 || len(rows[0]) != 32 || &rows[0][0] != &matrix.data[0] || &matrix.data[0] == old {
		t.Fatal("resized rows do not use the replacement slab")
	}
	for i, row := range matrix.rows[:cap(matrix.rows)] {
		if i >= len(rows) && row != nil {
			t.Fatalf("inactive row %d retains replaced storage", i)
		}
	}
	rows = matrix.resize(4, 8)
	for i, row := range rows {
		if len(row) != 8 || &row[0] != &matrix.data[i*8] {
			t.Fatalf("row %d was not rebuilt", i)
		}
	}
}

func TestVoxtralStreamingTailsOwnCompactStorage(t *testing.T) {
	cfg, w := buildTinyVoxtralRealtimeWeights()
	t.Run("convolution", func(t *testing.T) {
		conv := w.Encoder.Conv[1]
		rows := voxtralEncoderTestRows(0, 129, conv.In)
		var stream voxtralStreamConv
		if _, err := stream.push(context.Background(), conv, rows); err != nil {
			t.Fatal(err)
		}
		if len(stream.rows) == 0 || len(stream.rows) >= conv.Kernel {
			t.Fatalf("unexpected convolution tail: %d", len(stream.rows))
		}
		for i, row := range stream.rows {
			if cap(row) > len(stream.rows)*conv.In {
				t.Fatal("convolution tail retains a whole input chunk")
			}
			for j, v := range row {
				source := rows[len(rows)-len(stream.rows)+i]
				source[j] += 1
				if row[j] != v {
					t.Fatal("convolution tail aliases input chunk")
				}
			}
		}
	})
	t.Run("adapter", func(t *testing.T) {
		stream, err := newVoxtralStreamAudio(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := stream.push(context.Background(), w, make([]float32, 57), false); err != nil {
			t.Fatal(err)
		}
		if len(stream.residual) == 0 || len(stream.residual) >= cfg.Projector.DownsampleFactor {
			t.Fatalf("unexpected adapter tail: %d", len(stream.residual))
		}
		for _, row := range stream.residual {
			if cap(row) > len(stream.residual)*cfg.Encoder.DModel {
				t.Fatal("adapter tail retains a whole encoder chunk")
			}
		}
	})
}

type voxtralCancelInEncoder struct {
	context.Context
	cancel  context.CancelFunc
	session *VoxtralRealtimeSession
}

func (c *voxtralCancelInEncoder) Err() error {
	if c.session.audio != nil && c.session.audio.encoder.scratch.q.data != nil {
		c.cancel()
	}
	return c.Context.Err()
}

func TestVoxtralSessionDiscardsFailedEncoderWorkspace(t *testing.T) {
	cfg, w := buildTinyVoxtralRealtimeWeights()
	session := &VoxtralRealtimeSession{cfg: cfg, weights: w, disableFast: true}
	base, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := &voxtralCancelInEncoder{Context: base, cancel: cancel, session: session}
	if _, err := session.Push(ctx, nil, false); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-encoder cancellation: %v", err)
	}
	if !session.finished || session.audio == nil || session.audio.encoder.scratch.q.data == nil {
		t.Fatal("cancellation did not reach the allocated encoder workspace")
	}
	if _, err := session.Push(context.Background(), nil, false); !errors.Is(err, ErrVoxtralSessionClosed) {
		t.Fatalf("partially evaluated encoder was reused: %v", err)
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if session.audio != nil {
		t.Fatal("closed session retained encoder workspace")
	}
}

// This optional checkpoint test loads local weights only. Keeping one real
// block makes it useful on developer machines without expanding the entire
// 32-layer encoder just to verify workspace and layout parity.
func loadVoxtralEncoderWorkspaceModel(tb testing.TB) (VoxtralRealtimeConfig, VoxtralRealtimeWeights) {
	tb.Helper()
	path := os.Getenv("GOPHERLLM_VOXTRAL_MODEL")
	if path == "" {
		tb.Skip("set GOPHERLLM_VOXTRAL_MODEL to a local Voxtral Realtime GGUF")
	}
	mmap, err := OpenMmap(path)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { _ = mmap.Close() })
	gguf, err := ParseGGUF(mmap.Bytes())
	if err != nil {
		tb.Fatal(err)
	}
	// Load only block zero, borrowing its quantized bytes from the mapping.
	// Loading the entire decoder here would copy several GiB of unrelated
	// weights and distort this small encoder workspace benchmark's GC cost.
	tensors, inferred := indexTensors(gguf), inferTensorSizes(mmap.Bytes(), gguf)
	names := detectVoxtralTensorNames(tensors)
	n := names.encBlock(0)
	qInfo, gateInfo := tensors[n.Q], tensors[n.Gate]
	heads := int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.encoder.n_heads", "voxtral.encoder.n_heads"))
	if len(qInfo.Dims) != 2 || len(gateInfo.Dims) != 2 || heads <= 0 || int(qInfo.Dims[1])%heads != 0 {
		tb.Fatal("invalid encoder block dimensions")
	}
	cfg := VoxtralRealtimeConfig{RopeInterleaved: names.compact, Encoder: VoxtralRealtimeEncoderConfig{
		DModel: int(qInfo.Dims[0]), FFNDim: int(gateInfo.Dims[1]), NHeads: heads, NKVHeads: heads,
		HeadDim: int(qInfo.Dims[1]) / heads, NLayers: 1,
		SlidingWindow: int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.encoder.sliding_window", "voxtral.encoder.sliding_window")),
		RopeTheta:     gguFirstF32(gguf, 1e6, "stt.voxtral_realtime.encoder.rope_theta", "voxtral.encoder.rope_theta"),
		Epsilon:       gguFirstF32(gguf, 1e-5, "stt.voxtral_realtime.encoder.rms_norm_eps", "voxtral.encoder.norm_eps"),
	}}
	load := func(name string) Weight {
		weight, err := loadWeight(mmap.Bytes(), gguf.DataOffset, name, tensors, inferred, false, true, false, false)
		if err != nil {
			tb.Fatal(err)
		}
		return weight
	}
	loadVector := func(name string) []float32 {
		data, err := loadF32Vec(mmap.Bytes(), gguf.DataOffset, name, tensors, inferred)
		if err != nil {
			tb.Fatal(err)
		}
		return data
	}
	loadBias := func(name string, dim int) []float32 {
		return loadOptionalF32Vec(mmap.Bytes(), gguf.DataOffset, name, tensors, inferred, dim)
	}
	var w VoxtralRealtimeWeights
	w.Encoder.FinalNorm = loadVector(names.encFinalNorm())
	w.Encoder.Layers = []VoxtralRealtimeEncoderLayer{{
		AttnNorm: loadVector(n.AttnNorm), Q: load(n.Q), K: load(n.K), V: load(n.V), Out: load(n.Out),
		QB: loadBias(n.QB, int(qInfo.Dims[1])), VB: loadBias(n.VB, int(qInfo.Dims[1])), OutB: loadBias(n.OutB, cfg.Encoder.DModel),
		FFNNorm: loadVector(n.FFNNorm), FFNGate: load(n.Gate), FFNUp: load(n.Up), FFNDown: load(n.Down),
		FFNDownB: loadBias(n.DownB, cfg.Encoder.DModel),
	}}
	if err := prepareVoxtralStreamWeights(context.Background(), &w); err != nil {
		tb.Fatal(err)
	}
	return cfg, w
}

func TestVoxtralEncoderWorkspaceRealCheckpoint(t *testing.T) {
	cfg, w := loadVoxtralEncoderWorkspaceModel(t)
	checkVoxtralEncoderWorkspace(t, cfg, w, []int{2, 7, 1, 4})
}

func benchmarkVoxtralEncoderWorkspace(b *testing.B, cfg VoxtralRealtimeConfig, w VoxtralRealtimeWeights) {
	for _, n := range []int{1, 4, 16} {
		for _, reuse := range []bool{false, true} {
			mode := "channel_major_fresh"
			if reuse {
				mode = "rows_reused"
			}
			b.Run(mode+"/frames="+strconv.Itoa(n), func(b *testing.B) {
				ctx := context.Background()
				// Seed a valid zero-valued full history without benchmarking
				// an expensive offline prefill as part of setup. Both paths
				// therefore evaluate the same stable attention-window size.
				state := voxtralEncoderState{pos: cfg.Encoder.SlidingWindow, stride: cfg.Encoder.SlidingWindow + n}
				state.kHead = make([][]float32, len(w.Encoder.Layers))
				state.vHead = make([][]float32, len(w.Encoder.Layers))
				for li := range state.kHead {
					state.kHead[li] = make([]float32, cfg.Encoder.NHeads*state.stride*cfg.Encoder.HeadDim)
					state.vHead[li] = make([]float32, len(state.kHead[li]))
				}
				if _, err := forwardVoxtralEncoderRowsChunk(ctx, cfg, w, voxtralEncoderTestRows(0, n, cfg.Encoder.DModel), &state); err != nil {
					b.Fatal(err)
				}
				source := voxtralEncoderTestRows(1000, n, cfg.Encoder.DModel)
				rows := audioMatrixRows(n, cfg.Encoder.DModel)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for j := range rows {
						copy(rows[j], source[j])
					}
					var err error
					if reuse {
						_, err = forwardVoxtralEncoderRowsChunk(ctx, cfg, w, rows, &state)
					} else {
						state.scratch = voxtralEncoderScratch{}
						h := voxtralEncoderTestChannels(rows, cfg.Encoder.DModel)
						_, err = forwardVoxtralEncoderChunk(ctx, cfg, w, h, n, &state)
					}
					if err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// Compares the prior streaming handoff/workspace against the reused path at
// actual encoder widths. One layer keeps this a useful focused microbench.
func BenchmarkVoxtralEncoderStreamWorkspace(b *testing.B) {
	cfg, w := buildBenchVoxtralRealtimeWeights(1)
	benchmarkVoxtralEncoderWorkspace(b, cfg, w)
}

func BenchmarkVoxtralEncoderStreamCheckpoint(b *testing.B) {
	cfg, w := loadVoxtralEncoderWorkspaceModel(b)
	benchmarkVoxtralEncoderWorkspace(b, cfg, w)
}
