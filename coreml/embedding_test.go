package coreml

import (
	"context"
	"errors"
	"math"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

type tokenizeFunc func(string) []uint32

func (f tokenizeFunc) Encode(text string) []uint32 { return f(text) }

type fakePredictor struct {
	info    Info
	predict func([]int32, []int32, []int32, []float32) error
	calls   atomic.Int32
	closed  atomic.Int32
}

func (f *fakePredictor) Info() Info { return f.info }
func (f *fakePredictor) Predict(ids, mask, types []int32, output []float32) error {
	f.calls.Add(1)
	return f.predict(ids, mask, types, output)
}
func (f *fakePredictor) Close() error { f.closed.Add(1); return nil }

func TestEmbedBatchPaddingOwnershipAndScratch(t *testing.T) {
	fake := &fakePredictor{info: Info{BatchSize: 1, SequenceLength: 5, Dimension: 2}}
	var firstIDs, firstMask, firstTypes *int32
	var firstOutput *float32
	fake.predict = func(ids, mask, types []int32, out []float32) error {
		if !slices.Equal(types, []int32{0, 0, 0, 0, 0}) {
			t.Fatalf("token types = %v", types)
		}
		if fake.calls.Load() == 1 {
			if !slices.Equal(ids, []int32{3, 8, 9, 99, 99}) || !slices.Equal(mask, []int32{1, 1, 1, 0, 0}) {
				t.Fatalf("first padded input = %v/%v", ids, mask)
			}
			firstIDs, firstMask, firstTypes, firstOutput = &ids[0], &mask[0], &types[0], &out[0]
			copy(out, []float32{3, 4})
		} else {
			if !slices.Equal(ids, []int32{4, 99, 99, 99, 99}) || !slices.Equal(mask, []int32{1, 0, 0, 0, 0}) {
				t.Fatalf("short input retained previous tokens: %v/%v", ids, mask)
			}
			if &ids[0] != firstIDs || &mask[0] != firstMask || &types[0] != firstTypes || &out[0] != firstOutput {
				t.Fatal("prediction scratch was reallocated")
			}
			copy(out, []float32{4, 3})
		}
		return nil
	}
	e, err := newEmbedder(fake, tokenizeFunc(func(text string) []uint32 {
		if text == "long" {
			return []uint32{3, 8, 9}
		}
		return []uint32{4}
	}), Options{PadToken: 99, TokenTypeIDs: "token_type_ids"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = e.Close() })
	got, err := e.EmbedBatch(context.Background(), []string{"long", "short"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].TokenCount != 3 || got[1].TokenCount != 1 ||
		!slices.Equal(got[0].Embedding, []float32{.6, .8}) || !slices.Equal(got[1].Embedding, []float32{.8, .6}) {
		t.Fatalf("embeddings or input order = %+v", got)
	}
	got[1].Embedding[0] = 100
	if got[0].Embedding[0] != .6 {
		t.Fatal("results share mutable vector storage")
	}
}

func TestEmbedBatchFixedBatchTailAndReuse(t *testing.T) {
	const batch, sequence, dimension = 4, 3, 2
	fake := &fakePredictor{info: Info{BatchSize: batch, SequenceLength: sequence, Dimension: dimension}}
	var firstIDs *int32
	var firstOutput *float32
	fake.predict = func(ids, mask, types []int32, out []float32) error {
		if len(ids) != batch*sequence || len(out) != batch*dimension || len(types) != len(ids) {
			t.Fatalf("unexpected scratch sizes: %d/%d/%d", len(ids), len(out), len(types))
		}
		if firstIDs == nil {
			firstIDs, firstOutput = &ids[0], &out[0]
		} else if firstIDs != &ids[0] || firstOutput != &out[0] {
			t.Fatal("batch scratch was reallocated")
		}
		if !slices.Equal(types, make([]int32, batch*sequence)) {
			t.Fatal("token type input was not zero")
		}
		for row := range batch {
			base := row * sequence
			n := 0
			for column := range sequence {
				if mask[base+column] == 1 {
					n++
				} else if ids[base+column] != 99 {
					t.Fatalf("stale input in padded slot: %v/%v", ids, mask)
				}
			}
			if fake.calls.Load() > 1 && row > 0 &&
				(!slices.Equal(ids[base:base+sequence], ids[:sequence]) || !slices.Equal(mask[base:base+sequence], mask[:sequence])) {
				t.Fatalf("partial batch retained old rows instead of repeating a valid text: %v/%v", ids, mask)
			}
			// Unused output rows may be zero; the adapter must ignore them.
			out[row*dimension], out[row*dimension+1] = 0, 0
			if fake.calls.Load() == 1 || row == 0 {
				out[row*dimension], out[row*dimension+1] = float32(ids[base]), float32(n)
			}
		}
		return nil
	}
	e, err := newEmbedder(fake, tokenizeFunc(func(text string) []uint32 {
		n, _ := strconv.Atoi(text)
		return slices.Repeat([]uint32{uint32(n)}, 1+n%sequence)
	}), Options{PadToken: 99, TokenTypeIDs: "token_type_ids"})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	texts := []string{"1", "2", "3", "4", "5"}
	got, err := e.EmbedBatch(context.Background(), texts)
	if err != nil || len(got) != len(texts) || fake.calls.Load() != 2 {
		t.Fatalf("batched embedding = %v/%v, calls=%d", got, err, fake.calls.Load())
	}
	for i, row := range got {
		n, tokens := i+1, 1+(i+1)%sequence
		scale := math.Sqrt(float64(n*n + tokens*tokens))
		if row.TokenCount != tokens || math.Abs(float64(row.Embedding[0])-float64(n)/scale) > 1e-6 {
			t.Fatalf("batch row %d changed order or normalization: %+v", i, row)
		}
	}
	saved := slices.Clone(got[0].Embedding)
	if _, err := e.EmbedBatch(context.Background(), []string{"6"}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got[0].Embedding, saved) {
		t.Fatal("later predictions overwrote a returned embedding")
	}
	got[1].Embedding[0] = 100
	if !slices.Equal(got[0].Embedding, saved) {
		t.Fatal("batch outputs share mutable storage")
	}
	second := slices.Clone(got[1].Embedding)
	_ = append(got[0].Embedding, 123)
	if cap(got[0].Embedding) != len(got[0].Embedding) || !slices.Equal(got[1].Embedding, second) {
		t.Fatal("appending to one output overwrote the next output")
	}
}

func TestEmbedBatchValidatesEveryRealRow(t *testing.T) {
	fake := &fakePredictor{info: Info{BatchSize: 4, SequenceLength: 3, Dimension: 2}}
	e, err := newEmbedder(fake, tokenizeFunc(func(text string) []uint32 {
		if text == "invalid" {
			return []uint32{math.MaxUint32}
		}
		return []uint32{1}
	}), Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	got, err := e.EmbedBatch(context.Background(), []string{"valid", "valid", "invalid"})
	if got != nil || err == nil || !strings.Contains(err.Error(), "item 2 token 0") || fake.calls.Load() != 0 {
		t.Fatalf("invalid batch reached prediction: %v/%v, calls=%d", got, err, fake.calls.Load())
	}
}

func TestEmbedderRejectsUnboundedBatchShapes(t *testing.T) {
	for _, info := range []Info{
		{BatchSize: 0, SequenceLength: 3, Dimension: 2},
		{BatchSize: -1, SequenceLength: 3, Dimension: 2},
		{BatchSize: 9, SequenceLength: 3, Dimension: 2},
		{BatchSize: 4, SequenceLength: math.MaxInt, Dimension: 2},
		{BatchSize: 4, SequenceLength: 3, Dimension: math.MaxInt},
	} {
		fake := &fakePredictor{info: info}
		if _, err := newEmbedder(fake, tokenizeFunc(func(string) []uint32 { return []uint32{1} }), Options{}); err == nil {
			t.Fatalf("invalid model shape accepted: %+v", info)
		}
	}
}

func TestEmbedBatchRejectsInvalidTokens(t *testing.T) {
	for _, tokens := range [][]uint32{nil, {1, 2, 3, 4}, {math.MaxUint32}} {
		fake := &fakePredictor{info: Info{BatchSize: 1, SequenceLength: 3, Dimension: 2}}
		e, err := newEmbedder(fake, tokenizeFunc(func(string) []uint32 { return tokens }), Options{})
		if err != nil {
			t.Fatal(err)
		}
		got, err := e.EmbedBatch(context.Background(), []string{"bad input"})
		if err == nil || got != nil || fake.calls.Load() != 0 {
			t.Fatalf("tokens %v reached prediction: results=%v error=%v", tokens, got, err)
		}
		_ = e.Close()
	}
	if _, err := OpenEmbedder("unused", nil, Options{}); err == nil {
		t.Fatal("nil tokenizer accepted")
	}
	if _, err := OpenEmbedder("unused", tokenizeFunc(func(string) []uint32 { return nil }), Options{PadToken: math.MaxUint32}); err == nil {
		t.Fatal("padding ID outside int32 accepted")
	}
}

func TestEmbedBatchRejectsInvalidOutputAndNormalizesExtremes(t *testing.T) {
	for _, output := range [][]float32{{0, 0}, {float32(math.NaN()), 1}, {float32(math.Inf(1)), 1}, {1, float32(math.Inf(-1))}} {
		fake := &fakePredictor{info: Info{BatchSize: 1, SequenceLength: 1, Dimension: 2}, predict: func(_, _, _ []int32, out []float32) error {
			copy(out, output)
			return nil
		}}
		e, err := newEmbedder(fake, tokenizeFunc(func(string) []uint32 { return []uint32{1} }), Options{})
		if err != nil {
			t.Fatal(err)
		}
		got, err := e.EmbedBatch(context.Background(), []string{"invalid"})
		if err == nil || got != nil {
			t.Fatalf("invalid output %v accepted: %v/%v", output, got, err)
		}
		_ = e.Close()
	}
	for _, value := range []float32{math.MaxFloat32, 1e-40} {
		fake := &fakePredictor{info: Info{BatchSize: 1, SequenceLength: 1, Dimension: 2}, predict: func(_, _, _ []int32, out []float32) error {
			out[0], out[1] = value, value
			return nil
		}}
		e, err := newEmbedder(fake, tokenizeFunc(func(string) []uint32 { return []uint32{1} }), Options{})
		if err != nil {
			t.Fatal(err)
		}
		got, err := e.EmbedBatch(context.Background(), []string{"extreme"})
		if err != nil || math.Abs(float64(got[0].Embedding[0])-math.Sqrt(.5)) > 1e-6 {
			t.Fatalf("normalizing %g = %v/%v", value, got, err)
		}
		_ = e.Close()
	}
}

func TestEmbedBatchFailureDoesNotReturnPartialResults(t *testing.T) {
	want := errors.New("prediction failed")
	fake := &fakePredictor{info: Info{BatchSize: 1, SequenceLength: 1, Dimension: 2}}
	fake.predict = func(_, _, types []int32, out []float32) error {
		if types != nil {
			t.Error("disabled token type input was allocated")
		}
		if fake.calls.Load() == 2 {
			return want
		}
		copy(out, []float32{3, 4})
		return nil
	}
	e, err := newEmbedder(fake, tokenizeFunc(func(string) []uint32 { return []uint32{1} }), Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	got, err := e.EmbedBatch(context.Background(), []string{"first", "second"})
	if got != nil || !errors.Is(err, want) {
		t.Fatalf("failed batch = %v/%v", got, err)
	}
}

func TestEmbedderCancellationAndClose(t *testing.T) {
	started, unblock := make(chan struct{}), make(chan struct{})
	defer func() {
		select {
		case <-unblock:
		default:
			close(unblock)
		}
	}()
	fake := &fakePredictor{info: Info{BatchSize: 1, SequenceLength: 1, Dimension: 2}, predict: func(_, _, _ []int32, out []float32) error {
		close(started)
		<-unblock
		copy(out, []float32{3, 4})
		return nil
	}}
	e, err := newEmbedder(fake, tokenizeFunc(func(string) []uint32 { return []uint32{1} }), Options{})
	if err != nil {
		t.Fatal(err)
	}
	type result struct {
		values []gopherllm.EmbeddingResult
		err    error
	}
	activeCtx, cancelActive := context.WithCancel(context.Background())
	defer cancelActive()
	active := make(chan result, 1)
	go func() {
		values, err := e.EmbedBatch(activeCtx, []string{"active"})
		active <- result{values, err}
	}()
	<-started
	queuedCtx, cancelQueued := context.WithCancel(context.Background())
	queued := make(chan error, 1)
	go func() {
		_, err := e.EmbedBatch(queuedCtx, []string{"queued"})
		queued <- err
	}()
	cancelQueued()
	select {
	case err := <-queued:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("queued cancellation = %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("canceled caller waited for the active Core ML prediction")
	}
	closed := make(chan error, 1)
	go func() { closed <- e.Close() }()
	cancelActive()
	if fake.closed.Load() != 0 {
		t.Fatal("Close released the model during prediction")
	}
	close(unblock)
	got := <-active
	if got.values != nil || !errors.Is(got.err, context.Canceled) {
		t.Fatalf("active cancellation returned data: %+v", got)
	}
	if err := <-closed; err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil || fake.closed.Load() != 1 {
		t.Fatalf("model closed %d times: %v", fake.closed.Load(), err)
	}
	if _, err := e.EmbedBatch(context.Background(), []string{"after close"}); !errors.Is(err, ErrClosed) {
		t.Fatalf("embedding after Close = %v", err)
	}
	if fake.calls.Load() != 1 || e.Info().Dimension != 2 {
		t.Fatalf("unexpected prediction count or lost model info: %d/%+v", fake.calls.Load(), e.Info())
	}
}

// Isolate token preparation, output normalization and ownership from native
// model execution. Actual encoder latency is measured by the opt-in model test.
func BenchmarkEmbedBatchAdapter(b *testing.B) {
	for _, count := range []int{1, 32} {
		b.Run(strconv.Itoa(count), func(b *testing.B) {
			fake := &fakePredictor{info: Info{BatchSize: 4, SequenceLength: 64, Dimension: 768}}
			values := make([]float32, 4*768)
			for i := range values {
				values[i] = float32(i%17 - 8)
			}
			fake.predict = func(_, _, _ []int32, out []float32) error { copy(out, values); return nil }
			tokens := []uint32{1, 2, 3, 4, 5, 6, 7}
			e, err := newEmbedder(fake, tokenizeFunc(func(string) []uint32 { return tokens }), Options{})
			if err != nil {
				b.Fatal(err)
			}
			defer e.Close()
			texts := make([]string, count)
			b.ReportAllocs()
			for b.Loop() {
				rows, err := e.EmbedBatch(context.Background(), texts)
				if err != nil || len(rows) != count {
					b.Fatalf("embedding failed: %d/%v", len(rows), err)
				}
			}
		})
	}
}
