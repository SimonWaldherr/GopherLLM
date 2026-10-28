package engine

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

func qwen35MTPBatchFixture(quantized bool) (Config, Qwen35MTPWeights, Weight) {
	cfg, trunk := qwen35BatchFixture(quantized)
	cfg.LogitScale = 1
	rng := rand.New(rand.NewSource(3508))
	values := make([]float32, cfg.Dim*cfg.Dim*2)
	for i := range values {
		values[i] = float32(rng.NormFloat64()) * .035
	}
	eh := Weight{F32: values, Rows: cfg.Dim, Cols: 2 * cfg.Dim}
	if quantized {
		eh.F32, eh.Type = nil, GGMLTypeQ4_K
		for row := range cfg.Dim {
			eh.Raw = append(eh.Raw, QuantizeRowQ4K(values[row*eh.Cols:(row+1)*eh.Cols], eh.Cols)...)
		}
	}
	w := Qwen35MTPWeights{TokenEmbd: trunk.TokenEmbd, EmbeddingNorm: trunk.Layers[0].Norm,
		HiddenNorm: trunk.Layers[1].Norm, EHProj: eh, Norm: trunk.Layers[0].PostAttnNorm,
		PostAttnNorm: trunk.Layers[1].PostAttnNorm, Attention: trunk.Layers[1].Attention,
		FFN: trunk.Layers[1].FFN, SharedHeadNorm: trunk.OutputNorm,
		// An accidental vocabulary projection would panic for this empty raw
		// F32 tensor. Catch-up must leave it unused, including scalar tails.
		Output: Weight{Rows: cfg.VocabSize, Cols: cfg.Dim}}
	return cfg, w, trunk.Output
}

func qwen35MTPBatchHiddenRows(cfg Config, count int, buf *DecodeBuffer) [][]float32 {
	rows := reuseBatchViews(&buf.qwenBatch.XNFlat, &buf.qwenBatch.XN, count, cfg.Dim)
	rng := rand.New(rand.NewSource(3511))
	for _, row := range rows {
		for i := range row {
			row[i] = float32(rng.NormFloat64()) * .2
		}
	}
	return rows
}

func qwen35MTPFullCatchupReference(cfg Config, weights *Qwen35MTPWeights, state *Qwen35MTPState, token uint32, targetHidden []float32, pos int) {
	ForwardQwen35MTPBodyInto(cfg, *weights, state, token, state.PendingHidden, pos)
	copy(state.PendingHidden, targetHidden[:cfg.Dim])
}

func requireQwen35MTPBatchKV(t *testing.T, got, want *KVCache, positions int, quantized bool) {
	t.Helper()
	k, v := positions*got.PerPosKDim, positions*got.PerPosVDim
	switch got.kvFormat() {
	case kvF32:
		requireQwen35BatchValues(t, "MTP K", got.K[0][:k], want.K[0][:k])
		requireQwen35BatchValues(t, "MTP V", got.V[0][:v], want.V[0][:v])
	case kvF16:
		if quantized {
			if !slices.Equal(got.K16[0][:k], want.K16[0][:k]) || !slices.Equal(got.V16[0][:v], want.V16[0][:v]) {
				t.Fatal("quantized MTP catch-up changed F16 K/V")
			}
		} else {
			requireQwen35BatchF16(t, "MTP K", got.K16[0][:k], want.K16[0][:k])
			requireQwen35BatchF16(t, "MTP V", got.V16[0][:v], want.V16[0][:v])
		}
	case kvI8:
		kb, vb := positions*q8RowBytes(got.PerPosKDim), positions*q8RowBytes(got.PerPosVDim)
		if !slices.Equal(got.K8[0][:kb], want.K8[0][:kb]) || !slices.Equal(got.V8[0][:vb], want.V8[0][:vb]) {
			t.Fatal("quantized MTP catch-up changed I8 K/V")
		}
	}
}

func TestQwen35MTPBatchCatchupMatchesSerialAndPreservesTargetRows(t *testing.T) {
	for _, quantized := range []bool{false, true} {
		cfg, weights, output := qwen35MTPBatchFixture(quantized)
		for _, format := range []kvFormat{kvF32, kvF16, kvI8} {
			if format == kvI8 && !quantized {
				continue
			}
			for _, n := range []int{1, 2, 4, 7, 8, 31, 32, 33, 65} {
				t.Run(fmt.Sprintf("quant=%t/kv=%d/tokens=%d", quantized, format, n), func(t *testing.T) {
					got, want := newQwen35MTPState(cfg, cfg.MaxSeqLen, format), newQwen35MTPState(cfg, cfg.MaxSeqLen, format)
					targetBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
					hidden := qwen35MTPBatchHiddenRows(cfg, n+3, targetBuf)
					original := slices.Clone(targetBuf.qwenBatch.XNFlat)
					for pos := range 3 {
						qwen35MTPProcess(cfg, &weights, got, uint32(pos+1), hidden[pos], pos)
						qwen35MTPFullCatchupReference(cfg, &weights, want, uint32(pos+1), hidden[pos], pos)
					}
					tokens := make([]uint32, n)
					for i := range tokens {
						tokens[i] = uint32((i*7 + 5) % cfg.VocabSize)
						qwen35MTPFullCatchupReference(cfg, &weights, want, tokens[i], hidden[i+3], i+3)
					}
					got.Scratch.Logits = []float32{17, 23}
					if err := qwen35MTPProcessBatch(context.Background(), cfg, &weights, got, tokens, hidden[3:], 3); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(targetBuf.qwenBatch.XNFlat, original) || !slices.Equal(got.PendingHidden, hidden[n+2]) {
						t.Fatal("MTP catch-up changed borrowed target rows or used the wrong target boundary")
					}
					if !slices.Equal(got.Scratch.Logits, []float32{17, 23}) {
						t.Fatal("MTP catch-up evaluated the vocabulary head")
					}
					if len(got.Scratch.qwenBatch.X) > 32 {
						t.Fatal("MTP activation retention exceeded the 32-token chunk bound")
					}
					if quantized && !useQ8Activations.Load() && len(got.Scratch.qwenBatch.X) != 0 {
						t.Fatal("float-only quantized MTP backend entered batching")
					}
					requireQwen35MTPBatchKV(t, got.KV, want.KV, n+3, quantized)
					// A correct catch-up must remain usable for dependency-chained
					// drafting. Restore a real head only for this next operation.
					draftWeights := weights
					draftWeights.Output = output
					gd := slices.Clone(qwen35MTPDraftGreedy(cfg, &draftWeights, got, 4, n+3, 3, tokens, 1.1))
					wd := qwen35MTPDraftGreedy(cfg, &draftWeights, want, 4, n+3, 3, tokens, 1.1)
					if !slices.Equal(gd, wd) {
						t.Fatalf("drafting after batched catch-up=%v, want %v", gd, wd)
					}
					varied := false
					for i, logit := range got.Scratch.Logits {
						if !finite32(logit) {
							t.Fatal("draft comparison did not evaluate finite output logits")
						}
						varied = varied || i > 0 && logit != got.Scratch.Logits[0]
					}
					if !varied {
						t.Fatal("draft comparison used a constant vocabulary head")
					}
				})
			}
		}
	}
}

type qwen35MTPBatchCancelAfterWrite struct {
	context.Context
	cache  *KVCache
	before []float32
	cancel context.CancelFunc
}

func (c *qwen35MTPBatchCancelAfterWrite) Err() error {
	if !slices.Equal(c.cache.K[0], c.before) {
		c.cancel()
	}
	return c.Context.Err()
}

func TestQwen35MTPBatchCancellationAndValidation(t *testing.T) {
	cfg, weights, _ := qwen35MTPBatchFixture(false)
	state := newQwen35MTPState(cfg, cfg.MaxSeqLen, kvF32)
	targetBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
	hidden := qwen35MTPBatchHiddenRows(cfg, 4, targetBuf)
	tokens := []uint32{3, 4, 5, 6}
	pending := slices.Clone(state.PendingHidden)
	base, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := &qwen35MTPBatchCancelAfterWrite{Context: base, cache: state.KV, before: slices.Clone(state.KV.K[0]), cancel: cancel}
	if err := qwen35MTPProcessBatch(ctx, cfg, &weights, state, tokens, hidden, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("MTP catch-up ignored cancellation after a cache write: %v", err)
	}
	if slices.Equal(state.KV.K[0], ctx.before) || !slices.Equal(state.PendingHidden, pending) {
		t.Fatal("cancellation did not interrupt the partial attention chunk before publishing PendingHidden")
	}
	fresh := newQwen35MTPState(cfg, cfg.MaxSeqLen, kvF32)
	if err := qwen35MTPProcessBatch(ctx, cfg, &weights, fresh, tokens, hidden, 0); !errors.Is(err, context.Canceled) || !slices.Equal(fresh.KV.K[0], ctx.before) || len(fresh.Scratch.qwenBatch.X) != 0 {
		t.Fatal("pre-canceled MTP catch-up mutated the cache or initialized batch workspace")
	}
	for _, tc := range []struct {
		name   string
		start  int
		tokens []uint32
		hidden [][]float32
	}{
		{"negative offset", -1, tokens, hidden},
		{"cache overflow", cfg.MaxSeqLen - 1, tokens, hidden},
		{"missing rows", 0, tokens, hidden[:3]},
		{"short hidden", 0, tokens[:1], [][]float32{{1}}},
		{"invalid token", 0, []uint32{uint32(cfg.VocabSize)}, hidden[:1]},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := qwen35MTPProcessBatch(context.Background(), cfg, &weights, fresh, tc.tokens, tc.hidden, tc.start); err == nil || !slices.Equal(fresh.KV.K[0], ctx.before) || !slices.Equal(fresh.PendingHidden, pending) {
				t.Fatalf("invalid catch-up succeeded or modified state: %v", err)
			}
		})
	}
	if err := qwen35MTPProcessBatch(context.Background(), cfg, &weights, fresh, nil, nil, 0); err != nil {
		t.Fatal(err)
	}
}

func TestQwen35MTPBatchOptOutAndWorkspaceReuse(t *testing.T) {
	cfg, weights, _ := qwen35MTPBatchFixture(false)
	state := newQwen35MTPState(cfg, cfg.MaxSeqLen, kvF32)
	targetBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
	hidden := qwen35MTPBatchHiddenRows(cfg, 8, targetBuf)
	tokens := []uint32{3, 4, 5, 6, 7, 8, 9, 10}
	t.Setenv("GOPHERLLM_NO_MTP_BATCH", "1")
	if err := qwen35MTPProcessBatch(context.Background(), cfg, &weights, state, tokens, hidden, 0); err != nil || len(state.Scratch.qwenBatch.X) != 0 {
		t.Fatalf("MTP batch opt-out was ignored: %v", err)
	}
	t.Setenv("GOPHERLLM_NO_MTP_BATCH", "")
	state.reset()
	if err := qwen35MTPProcessBatch(context.Background(), cfg, &weights, state, tokens, hidden, 0); err != nil {
		t.Fatal(err)
	}
	b := &state.Scratch.qwenBatch
	flats := func() [][]float32 {
		return [][]float32{b.XFlat, b.XNFlat, b.QKVFlat, b.GateFlat, b.KFlat, b.VFlat}
	}
	before := flats()
	state.reset()
	if err := qwen35MTPProcessBatch(context.Background(), cfg, &weights, state, tokens, hidden, 0); err != nil {
		t.Fatal(err)
	}
	for i, flat := range flats() {
		if len(flat) != len(before[i]) || &flat[0] != &before[i][0] {
			t.Fatalf("MTP batch activation slab %d did not reuse its backing storage", i)
		}
	}
}

func TestQwen35MTPBatchSkipsUnusedBodyProjections(t *testing.T) {
	cfg, weights, _ := qwen35MTPBatchFixture(false)
	original := weights
	// Invalid empty matrices panic if the unused Q/gate, O or FFN path is
	// accidentally evaluated. K/V must still match the full valid MTP graph.
	weights.Attention.Q, weights.Attention.O = Weight{}, Weight{}
	weights.FFN.Gate, weights.FFN.Up, weights.FFN.Down = Weight{}, Weight{}, Weight{}
	for _, n := range []int{1, 8} {
		got, want := newQwen35MTPState(cfg, cfg.MaxSeqLen, kvF32), newQwen35MTPState(cfg, cfg.MaxSeqLen, kvF32)
		targetBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
		hidden := qwen35MTPBatchHiddenRows(cfg, n, targetBuf)
		tokens := make([]uint32, n)
		for i := range tokens {
			tokens[i] = uint32(i % cfg.VocabSize)
			qwen35MTPFullCatchupReference(cfg, &original, want, tokens[i], hidden[i], i)
		}
		if err := qwen35MTPProcessBatch(context.Background(), cfg, &weights, got, tokens, hidden, 0); err != nil {
			t.Fatal(err)
		}
		requireQwen35MTPBatchKV(t, got.KV, want.KV, n, false)
		if !slices.Equal(got.PendingHidden, want.PendingHidden) {
			t.Fatal("omitting unused MTP body projections changed the target boundary")
		}
	}
}

func TestQwen35MTPPrivateBodyPreservesCPUWrapperAndKV(t *testing.T) {
	cfg, weights, _ := qwen35MTPBatchFixture(false)
	// F32 matvecs ignore Metal views. This harmless marker selects the private
	// mixed Q/O/FFN graph without requiring a GPU in the portable unit test.
	weights.Attention.Q.Metal = new(MetalWeight)
	got, want := newQwen35MTPState(cfg, 8, kvF32), newQwen35MTPState(cfg, 8, kvF32)
	targetBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
	hidden := qwen35MTPBatchHiddenRows(cfg, 4, targetBuf)
	for pos := range hidden {
		token := uint32(pos + 3)
		ForwardQwen35MTPBodyInto(cfg, weights, want, token, hidden[pos], pos)
		forwardQwen35MTPBodyInto(cfg, weights, got, token, hidden[pos], pos)
		if !slices.Equal(got.KV.K[0], want.KV.K[0]) || !slices.Equal(got.KV.V[0], want.KV.V[0]) {
			t.Fatal("explicit Q routing changed the CPU K/V contract")
		}
		requireQwen35BatchValues(t, "private full MTP hidden", got.Scratch.XN, want.Scratch.XN)
	}
}

func TestQwen35MTPKVOnlyRejectsWebGPUProjectionViews(t *testing.T) {
	previousQ8 := useQ8Activations.Load()
	t.Cleanup(func() { useQ8Activations.Store(previousQ8) })
	for _, q8 := range []bool{false, true} {
		useQ8Activations.Store(q8)
		for _, field := range []string{"EH", "Q", "K", "V", "embedding", "O", "FFN", "output"} {
			t.Run(fmt.Sprintf("Q8=%t/%s", q8, field), func(t *testing.T) {
				_, w, _ := qwen35MTPBatchFixture(false)
				dummy := new(GPUWeight)
				switch field {
				case "EH":
					w.EHProj.GPU = dummy
				case "Q":
					w.Attention.Q.GPU = dummy
				case "K":
					w.Attention.K.GPU = dummy
				case "V":
					w.Attention.V.GPU = dummy
				case "embedding":
					w.TokenEmbd.GPU = dummy
				case "O":
					w.Attention.O.GPU = dummy
				case "FFN":
					w.FFN.Gate.GPU, w.FFN.Up.GPU, w.FFN.Down.GPU = dummy, dummy, dummy
				case "output":
					w.Output.GPU = dummy
				}
				want := field != "EH" && field != "Q" && field != "K" && field != "V"
				if qwen35MTPKVOnlyEligible(w) != want || qwen35MTPBatchNumerics(w) != want {
					t.Fatalf("WebGPU %s: KV-only and batch gates must both be %t", field, want)
				}
			})
		}
	}
}

func TestQwen35MTPWebGPUViewsUseFullBodyCatchup(t *testing.T) {
	for _, field := range []string{"EH", "Q", "K", "V", "mixed QKV"} {
		for _, n := range []int{1, 4} {
			t.Run(fmt.Sprintf("%s/tokens=%d", field, n), func(t *testing.T) {
				quantized := field == "mixed QKV"
				cfg, w, _ := qwen35MTPBatchFixture(quantized)
				dummy := new(GPUWeight)
				switch field {
				case "EH":
					w.EHProj.GPU = dummy
				case "Q":
					w.Attention.Q.GPU = dummy
				case "K":
					w.Attention.K.GPU = dummy
				case "V":
					w.Attention.V.GPU = dummy
				case "mixed QKV":
					// The production Q4_K/Q4_K/Q6_K triple is fused on the
					// CPU even with GPU views. Dummy handles are never used.
					w.Attention.Q.GPU, w.Attention.K.GPU, w.Attention.V.GPU = dummy, dummy, dummy
				}
				// F32 cases ignore their dummy GPU views before any dispatch;
				// the quantized case above uses the CPU Q/K/V fusion.
				got, want := newQwen35MTPState(cfg, 8, kvF32), newQwen35MTPState(cfg, 8, kvF32)
				targetBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
				hidden := qwen35MTPBatchHiddenRows(cfg, n, targetBuf)
				tokens := make([]uint32, n)
				for i := range tokens {
					tokens[i] = uint32(i + 3)
					qwen35MTPFullCatchupReference(cfg, &w, want, tokens[i], hidden[i], i)
				}
				if err := qwen35MTPProcessBatch(context.Background(), cfg, &w, got, tokens, hidden, 0); err != nil {
					t.Fatal(err)
				}
				if len(got.Scratch.qwenBatch.X) != 0 || len(got.Scratch.QGate) == 0 {
					t.Fatal("WebGPU catch-up did not retain the full scalar attention body")
				}
				requireQwen35MTPBatchKV(t, got.KV, want.KV, n, quantized)
				if !slices.Equal(got.Scratch.XN, want.Scratch.XN) || !slices.Equal(got.PendingHidden, hidden[n-1]) {
					t.Fatal("WebGPU fallback differs from the independent full-body catch-up")
				}
			})
		}
	}
}

func BenchmarkQwen35MTPBatchCatchup(b *testing.B) {
	cfg, weights, _ := qwen35MTPBatchFixture(true)
	for _, n := range []int{1, 2, 4, 8, 32} {
		for _, mode := range []string{"full_body", "kv_scalar", "kv_batch"} {
			b.Run(fmt.Sprintf("tokens%d/%s", n, mode), func(b *testing.B) {
				state := newQwen35MTPState(cfg, cfg.MaxSeqLen, kvF32)
				targetBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
				hidden := qwen35MTPBatchHiddenRows(cfg, n, targetBuf)
				tokens := make([]uint32, n)
				for i := range tokens {
					tokens[i] = uint32((i*7 + 3) % cfg.VocabSize)
				}
				run := func() {
					state.reset()
					if mode == "kv_batch" {
						if err := qwen35MTPProcessBatch(context.Background(), cfg, &weights, state, tokens, hidden, 0); err != nil {
							b.Fatal(err)
						}
					} else {
						for i, token := range tokens {
							if mode == "full_body" {
								qwen35MTPFullCatchupReference(cfg, &weights, state, token, hidden[i], i)
							} else {
								qwen35MTPProcess(cfg, &weights, state, token, hidden[i], i)
							}
						}
					}
				}
				run()
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					run()
				}
			})
		}
	}
}
