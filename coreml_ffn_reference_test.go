//go:build coreml

package gopherllm

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// The original token-at-a-time reference is kept here to check bitwise parity
// and measure the cost of decoding the same weight rows for each probe.
func serialCoreMLFFNReference(m coreMLFFNManifest, gate, up, down Weight, input []float32) []float32 {
	probes := min(m.Tokens, 3)
	result := make([]float32, probes*m.Dimension)
	gates, ups, hidden := make([]float32, m.HiddenDimension), make([]float32, m.HiddenDimension), make([]float32, m.HiddenDimension)
	var row []float32
	project := func(w Weight, x, out []float32) {
		for i := range out {
			w.RowInto(i, len(x), &row)
			out[i] = DotF32(row, x)
		}
	}
	for token := 0; token < probes; token++ {
		x := input[token*m.Dimension : (token+1)*m.Dimension]
		project(gate, x, gates)
		project(up, x, ups)
		siluMulF32(gates, ups, hidden)
		project(down, hidden, result[token*m.Dimension:(token+1)*m.Dimension])
	}
	return result
}

func coreMLFFNReferenceWeight(rng *rand.Rand, rows, cols int, typ GGMLType) Weight {
	w := Weight{Rows: rows, Cols: cols, Type: typ}
	if typ == GGMLTypeF32 {
		w.F32 = randomVec(rng, rows*cols)
		return w
	}
	size, ok := typ.DataSize(rows * cols)
	if !ok {
		panic("unsupported reference weight type")
	}
	w.Raw = make([]byte, size)
	if typ == GGMLTypeF16 {
		for i := 0; i < size; i += 2 {
			binary.LittleEndian.PutUint16(w.Raw[i:], F32ToF16((rng.Float32()-.5)*.125))
		}
		return w
	}
	rowSize := size / rows
	for i := 0; i < rows; i++ {
		var row []byte
		switch typ {
		case GGMLTypeQ4_K:
			row = randomQ4KRow(rng, cols)
		case GGMLTypeQ6_K:
			row = randomQ6KRow(rng, cols)
		default:
			panic("unsupported reference weight type")
		}
		copy(w.Raw[i*rowSize:], row)
	}
	return w
}

func TestCoreMLFFNReferenceMatchesSerial(t *testing.T) {
	for _, shape := range []struct {
		dim, hidden int
		gate, down  GGMLType
	}{
		{7, 15, GGMLTypeF32, GGMLTypeF32},
		{17, 127, GGMLTypeF32, GGMLTypeF32},
		{17, 128, GGMLTypeF32, GGMLTypeF32},
		{17, 129, GGMLTypeF16, GGMLTypeF16},
		{23, 257, GGMLTypeF16, GGMLTypeF16},
		{256, 512, GGMLTypeQ4_K, GGMLTypeQ6_K},
	} {
		rng := rand.New(rand.NewSource(81734))
		gate := coreMLFFNReferenceWeight(rng, shape.hidden, shape.dim, shape.gate)
		up := coreMLFFNReferenceWeight(rng, shape.hidden, shape.dim, shape.gate)
		down := coreMLFFNReferenceWeight(rng, shape.dim, shape.hidden, shape.down)
		for _, tokens := range []int{1, 2, 3, 5, 32} {
			t.Run(fmt.Sprintf("type=%d/dim=%d/hidden=%d/tokens=%d", shape.gate, shape.dim, shape.hidden, tokens), func(t *testing.T) {
				m := coreMLFFNManifest{Tokens: tokens, Dimension: shape.dim, HiddenDimension: shape.hidden}
				input := randomVec(rng, tokens*shape.dim)
				want := serialCoreMLFFNReference(m, gate, up, down, input)
				got := coreMLFFNReference(m, gate, up, down, input)
				if len(got) != min(tokens, 3)*shape.dim {
					t.Fatalf("wrong reference length: %d", len(got))
				}
				for i := range want {
					if !finite32(want[i]) || math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("reference[%d]=%g (%08x), want %g (%08x)", i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
					}
				}
			})
		}
	}
}

func TestCoreMLFFNQualityChecksEveryProbe(t *testing.T) {
	r, m, _ := coreMLFFNManifestFixture(t)
	m.Tokens = 5
	gate, up, down, err := r.coreMLFFNWeights(0)
	if err != nil {
		t.Fatal(err)
	}
	for bad := -1; bad < 3; bad++ {
		t.Run(fmt.Sprint(bad), func(t *testing.T) {
			fake := fakeCoreMLFFNForSource(t, r, m)
			predict := fake.predict
			fake.predict = func(input, out []float32) error {
				if err := predict(input, out); err != nil {
					return err
				}
				if bad >= 0 {
					for i := bad * m.Dimension; i < (bad+1)*m.Dimension; i++ {
						out[i] = -out[i]
					}
				}
				return nil
			}
			if err := validateCoreMLFFNQuality(fake, m, gate, up, down); (err != nil) != (bad >= 0) {
				t.Fatalf("bad probe %d: %v", bad, err)
			}
		})
	}
}

var coreMLFFNReferenceResult []float32

func BenchmarkCoreMLFFNReference(b *testing.B) {
	// Synthetic Q4_K gate/up and Q6_K down tensors with real FFN dimensions.
	// No Core ML prediction, source hashing or model load is timed here.
	for _, shape := range []struct {
		name        string
		dim, hidden int
	}{{"ministral3b", 3072, 9216}, {"qwen27b", 5120, 17408}} {
		rng := rand.New(rand.NewSource(81734))
		gate := coreMLFFNReferenceWeight(rng, shape.hidden, shape.dim, GGMLTypeQ4_K)
		up := coreMLFFNReferenceWeight(rng, shape.hidden, shape.dim, GGMLTypeQ4_K)
		down := coreMLFFNReferenceWeight(rng, shape.dim, shape.hidden, GGMLTypeQ6_K)
		m := coreMLFFNManifest{Tokens: 3, Dimension: shape.dim, HiddenDimension: shape.hidden}
		input := randomVec(rng, m.Tokens*m.Dimension)
		for _, mode := range []string{"serial", "shared"} {
			b.Run(shape.name+"/"+mode, func(b *testing.B) {
				project := serialCoreMLFFNReference
				if mode == "shared" {
					project = coreMLFFNReference
				}
				coreMLFFNReferenceResult = project(m, gate, up, down, input)
				b.ReportAllocs()
				for b.Loop() {
					coreMLFFNReferenceResult = project(m, gate, up, down, input)
				}
			})
		}
	}
}
