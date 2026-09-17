//go:build darwin && cgo && metal

package gopherllm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

func TestMetalGeGLUParity(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	t.Setenv("GOPHERLLM_METAL_GEMMA", "1")
	rng := rand.New(rand.NewSource(634))
	makeWeight := func(rows, cols int, kind GGMLType) Weight {
		var raw []byte
		for range rows {
			var row []byte
			switch kind {
			case GGMLTypeQ4_K:
				row = randomQ4KRow(rng, cols)
				for b := 0; b < len(row); b += 144 {
					row[b+1] = 0x08
					row[b+3] = 0x04
				}
			case GGMLTypeQ6_K:
				row = randomQ6KRow(rng, cols)
				for b := 0; b < len(row); b += 210 {
					row[b+209] = 0x04
				}
			case GGMLTypeQ8_0:
				row = randomQ8_0Row(rng, cols)
				for b := 0; b < len(row); b += 34 {
					row[b+1] = 0x18
				}
			}
			raw = append(raw, row...)
		}
		m := &MetalWeight{typ: kind, rows: rows, cols: cols}
		switch kind {
		case GGMLTypeQ4_K:
			m.q4 = metalbackend.PrepareQ4K(raw, rows, cols, false)
		case GGMLTypeQ6_K:
			m.q6 = metalbackend.PrepareQ6K(raw, rows, cols, false)
		case GGMLTypeQ8_0:
			m.q8 = metalbackend.PrepareQ8_0(raw, rows, cols, false)
		}
		t.Cleanup(func() { releaseMetalWeight(m) })
		return Weight{Raw: raw, Rows: rows, Cols: cols, Type: kind, Metal: m}
	}
	for _, types := range [][3]GGMLType{{GGMLTypeQ4_K, GGMLTypeQ4_K, GGMLTypeQ6_K}, {GGMLTypeQ4_K, GGMLTypeQ6_K, GGMLTypeQ4_K}, {GGMLTypeQ8_0, GGMLTypeQ8_0, GGMLTypeQ8_0}} {
		g, u, d := makeWeight(768, 256, types[0]), makeWeight(768, 256, types[1]), makeWeight(257, 768, types[2])
		cg, cu, cd := g, u, d
		cg.Metal = nil
		cu.Metal = nil
		cd.Metal = nil
		for _, batch := range []int{1, 3, 16, 33} {
			t.Run(fmt.Sprintf("%v/batch%d", types, batch), func(t *testing.T) {
				x := make([]float32, batch*256)
				for i := range x {
					x[i] = float32(rng.NormFloat64()) * .2
				}
				var got, want []float32
				if !matvecMetalGeGLUInto(g, u, d, x, batch, &got) {
					t.Fatal("fusion unavailable", MetalError())
				}
				for i := 0; i < batch; i++ {
					var a, b, h, o []float32
					cg.MatvecInto(x[i*256:(i+1)*256], &a)
					cu.MatvecInto(x[i*256:(i+1)*256], &b)
					h = make([]float32, len(a))
					geluMulF32(a, b, h)
					cd.MatvecInto(h, &o)
					want = append(want, o...)
				}
				assertMetalFiniteClose(t, got, want)
			})
		}
		input := metalTestVector(d.Cols)
		var projected, reference []float32
		cd.MatvecInto(input, &reference)
		if !matvecMetalGemmaOutputInto(d, input, &projected) {
			t.Fatal("output unavailable")
		}
		assertMetalFiniteClose(t, projected, reference)
		recent := []uint32{1, 1, 3, 999999}
		applyRepeatPenalty(reference, recent, 1.1)
		next, ok := argmaxMetalGemmaOutput(d, input, recent, 1.1)
		if !ok || next != argmaxFiniteToken(reference) {
			t.Fatal("greedy output mismatch")
		}
		for i := range input {
			input[i] = float32(math.NaN())
		}
		if _, ok := argmaxMetalGemmaOutput(d, input, nil, 1); ok {
			t.Fatal("nonfinite greedy result accepted")
		}
		var out []float32
		if matvecMetalGeGLUInto(g, u, d, make([]float32, 257*256), 257, &out) {
			t.Fatal("oversized batch accepted")
		}
		if matvecMetalGeGLUInto(cg, u, d, make([]float32, 256), 1, &out) {
			t.Fatal("unprepared gate accepted")
		}
	}
}
