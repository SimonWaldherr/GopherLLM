package engine

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type qi21TransformerReference struct {
	Cases []struct {
		Height, Width int
		Text          []float32
		Steps         []struct {
			Time             float32
			Latent, Expected []float32
		}
	}
	Schedules []struct {
		Tokens, Steps int
		Expected      []float32
	}
}

func qi21TransformerFixture(t *testing.T) (*qi21Transformer, qi21TransformerReference) {
	t.Helper()
	dir := filepath.Join("../../testdata", "qwen_image21_transformer")
	var cfg qi21TransformerConfig
	if err := readModelJSON(filepath.Join(dir, "config.json"), 64<<10, &cfg); err != nil {
		t.Fatal(err)
	}
	source, err := qi21OpenWeights(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(source.close)
	model, err := qi21NewTransformer(context.Background(), source, cfg)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref qi21TransformerReference
	if err = json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	return model, ref
}

func qi21AssertNear(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d, expected %d", len(got), len(want))
	}
	for i, v := range got {
		if !finite32(v) || math.Abs(float64(v-want[i])) > tolerance {
			t.Fatalf("element %d = %.9g, expected %.9g (delta %.9g)", i, v, want[i], v-want[i])
		}
	}
}

func TestQwenImage21TransformerReference(t *testing.T) {
	model, ref := qi21TransformerFixture(t)
	for _, tc := range ref.Cases {
		cached, err := model.newState(tc.Text, tc.Height, tc.Width)
		if err != nil {
			t.Fatal(err)
		}
		for _, step := range tc.Steps {
			got, err := cached.predict(context.Background(), step.Latent, step.Time)
			if err != nil {
				t.Fatal(err)
			}
			qi21AssertNear(t, got, step.Expected, 2e-5)
			uncached, err := model.newState(tc.Text, tc.Height, tc.Width)
			if err != nil {
				t.Fatal(err)
			}
			fresh, err := uncached.predict(context.Background(), step.Latent, step.Time)
			if err != nil {
				t.Fatal(err)
			}
			qi21AssertNear(t, got, fresh, 2e-6)
			for _, kv := range cached.cache {
				want := cached.prefix * model.dim
				if len(kv.k) != want || cap(kv.k) != want || len(kv.v) != want || cap(kv.v) != want {
					t.Fatal("prefix cache retained image-token storage")
				}
				if &kv.k[0] == &cached.k[0] || &kv.v[0] == &cached.v[0] {
					t.Fatal("prefix cache aliases mutable projection scratch")
				}
			}
		}
	}
}

func TestQwenImage21TransformerValidationAndCancellation(t *testing.T) {
	model, ref := qi21TransformerFixture(t)
	badConfig := model.cfg
	maxEven := int(^uint(0)>>1) - 1
	// The sum would overflow back to HeadDim without individual axis bounds.
	badConfig.AxesDims = []int{maxEven, maxEven, model.cfg.HeadDim + 4}
	if _, err := qi21NewTransformer(context.Background(), nil, badConfig); err == nil {
		t.Fatal("accepted overflowing rotary dimensions")
	}
	tc := ref.Cases[0]
	step := tc.Steps[0]
	if _, err := model.newState([]float32{1}, 2, 2); err == nil {
		t.Fatal("accepted incomplete context vector")
	}
	badText := slices.Clone(tc.Text)
	badText[0] = float32(math.NaN())
	if _, err := model.newState(badText, 2, 2); err == nil {
		t.Fatal("accepted non-finite context")
	}
	state, err := model.newState(tc.Text, tc.Height, tc.Width)
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range []float32{-1, 1.1, float32(math.NaN())} {
		if _, err := state.predict(context.Background(), step.Latent, bad); err == nil {
			t.Fatal("accepted invalid timestep")
		}
	}
	if _, err := state.predict(context.Background(), step.Latent[:len(step.Latent)-1], .5); err == nil {
		t.Fatal("accepted incomplete latent")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := state.predict(ctx, step.Latent, .5); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	if state.valid {
		t.Fatal("cancelled first step published its cache")
	}
	got, err := state.predict(context.Background(), step.Latent, step.Time)
	if err != nil {
		t.Fatal(err)
	}
	qi21AssertNear(t, got, step.Expected, 2e-5)
}

func TestQwenImage21TransformerStateMemoryBudget(t *testing.T) {
	// Projection weights are irrelevant to admission. A one-channel prompt
	// avoids allocating large test embeddings while exercising real DiT widths.
	m := &qi21Transformer{
		cfg: qi21TransformerConfig{ContextDim: 1, OutChannels: 64, HeadDim: 128, MLPRatio: 3, AxesDims: []int{16, 56, 56}},
		dim: 4096, layers: make([]qi21TransformerLayer, 32),
	}
	if _, err := m.newState(make([]float32, 64), 16, 16); err != nil {
		t.Fatalf("short prompt at 256x256: %v", err)
	}
	if _, err := m.newState(make([]float32, 4096), 16, 16); err == nil || !strings.Contains(err.Error(), "exceeds 2 GiB") {
		t.Fatalf("oversized prefix cache = %v", err)
	}
	m.dim = 16384
	if _, err := m.newState(make([]float32, 64), 128, 128); err == nil || !strings.Contains(err.Error(), "exceeds 2 GiB") {
		t.Fatalf("oversized image workspace = %v", err)
	}
}

func TestQwenImage21RoPEGeometry(t *testing.T) {
	axes := []int{4, 6, 6}
	prefix, height, width := 3, 3, 4
	cosine, sine := qi21TextImageRoPE(axes, prefix, height, width)
	if len(cosine) != (prefix+height*width)*8 {
		t.Fatal("wrong rotary dimensions")
	}
	// Text advances all axes. Image time stays at prefix; its grid is centered
	// independently of the prefix, including the asymmetric odd-height range.
	for _, check := range []struct {
		token, pair int
		angle       float64
	}{{1, 0, 1}, {1, 2, 1}, {1, 5, 1}, {3, 0, 3}, {3, 2, -2}, {3, 5, -2}, {14, 0, 3}, {14, 2, 0}, {14, 5, 1}} {
		i := check.token*8 + check.pair
		if math.Abs(float64(cosine[i])-math.Cos(check.angle)) > 1e-6 || math.Abs(float64(sine[i])-math.Sin(check.angle)) > 1e-6 {
			t.Fatalf("bad rotary position: %+v", check)
		}
	}
}

type qi21QuantizedTestSource struct {
	source   qi21WeightSource
	expanded bool
}

func (s qi21QuantizedTestSource) floats(name string, shape ...int) ([]float32, error) {
	return s.source.floats(name, shape...)
}
func (s qi21QuantizedTestSource) matrix(name string, rows, cols int) (Weight, error) {
	w, err := s.source.matrix(name, rows, cols)
	if err != nil || cols%32 != 0 {
		return w, err
	}
	raw := make([]byte, 0, rows*(cols/32)*34)
	row := make([]float32, cols)
	for i := 0; i < rows; i++ {
		w.RowInto(i, cols, &row)
		raw = append(raw, QuantizeRowQ8_0(row, cols)...)
	}
	w = Weight{Raw: raw, Type: GGMLTypeQ8_0, Rows: rows, Cols: cols}
	if s.expanded {
		values := make([]float32, rows*cols)
		for i := 0; i < rows; i++ {
			w.RowInto(i, cols, &row)
			copy(values[i*cols:], row)
		}
		w = Weight{F32: values, Type: GGMLTypeF32, Rows: rows, Cols: cols}
	}
	return w, nil
}

func TestQwenImage21TransformerPackedWeights(t *testing.T) {
	model, ref := qi21TransformerFixture(t)
	source, err := qi21OpenWeights(context.Background(), filepath.Join("../../testdata", "qwen_image21_transformer"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.close()
	packed, err := qi21NewTransformer(context.Background(), qi21QuantizedTestSource{source: source}, model.cfg)
	if err != nil {
		t.Fatal(err)
	}
	expanded, err := qi21NewTransformer(context.Background(), qi21QuantizedTestSource{source: source, expanded: true}, model.cfg)
	if err != nil {
		t.Fatal(err)
	}
	tc := ref.Cases[2]
	a, err := packed.newState(tc.Text, tc.Height, tc.Width)
	if err != nil {
		t.Fatal(err)
	}
	b, err := expanded.newState(tc.Text, tc.Height, tc.Width)
	if err != nil {
		t.Fatal(err)
	}
	for _, step := range tc.Steps {
		got, err := a.predict(context.Background(), step.Latent, step.Time)
		if err != nil {
			t.Fatal(err)
		}
		want, err := b.predict(context.Background(), step.Latent, step.Time)
		if err != nil {
			t.Fatal(err)
		}
		qi21AssertNear(t, got, want, 2e-5)
	}
}

func TestQwenImage21ProjectionBoundaries(t *testing.T) {
	for _, cols := range []int{64, 8192, 12288, 16384} {
		const rows = 32
		values := make([]float32, rows*cols)
		raw := make([]byte, 0, rows*cols/32*34)
		for r := 0; r < rows; r++ {
			row := values[r*cols : (r+1)*cols]
			for j := range row {
				row[j] = float32((r+j)%19-9) / 32
			}
			raw = append(raw, QuantizeRowQ8_0(row, cols)...)
		}
		w := Weight{Raw: raw, Type: GGMLTypeQ8_0, Rows: rows, Cols: cols}
		decoded := make([]float32, cols)
		for _, tokens := range []int{129, 257, 288, 385} {
			input, output := make([]float32, tokens*cols), make([]float32, tokens*rows)
			for i := range input {
				input[i] = float32(i%23-11) / 64
			}
			if err := qi21Project(context.Background(), w, input, output); err != nil {
				t.Fatal(err)
			}
			// Sample every possible boundary of the old and balanced layouts
			// against a scalar F64 oracle using the same decoded Q8 weights.
			for _, token := range []int{0, 127, 128, 143, 144, 255, 256, tokens - 1} {
				if token >= tokens {
					continue
				}
				for r := 0; r < rows; r++ {
					w.RowInto(r, cols, &decoded)
					var want float64
					for j, v := range decoded {
						want += float64(v) * float64(input[token*cols+j])
					}
					got := float64(output[token*rows+r])
					if math.Abs(got-want) > 1e-4+math.Abs(want)*1e-4 {
						t.Fatalf("cols=%d tokens=%d token=%d row=%d: %g, want %g", cols, tokens, token, r, got, want)
					}
				}
			}
		}
	}
}
