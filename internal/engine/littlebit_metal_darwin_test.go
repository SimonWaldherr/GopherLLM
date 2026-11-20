//go:build darwin && cgo && metal

package engine

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

func littleBitMetalTestWeight(rows, cols, rank, residualRank int, bias bool) Weight {
	random := rand.New(rand.NewSource(15))
	matrix := func(rows, cols int) []byte {
		words := (cols + 31) / 32
		data := make([]byte, rows*words*4)
		for row := range rows {
			for word := range words {
				signs := random.Uint32()
				if unused := words*32 - cols; word == words-1 && unused != 0 {
					signs >>= unused
				}
				binary.LittleEndian.PutUint32(data[4*(row*words+word):], signs)
			}
		}
		return data
	}
	scales := func(n int, typ GGMLType) LittleBitScale {
		size := 2
		if typ == GGMLTypeF32 {
			size = 4
		}
		data := make([]byte, n*size)
		for i := range n {
			fraction := random.Intn(16)
			value := float32(fraction+8) / 16
			if i%3 == 0 {
				value = -value
			}
			bits := math.Float32bits(value)
			switch typ {
			case GGMLTypeF32:
				binary.LittleEndian.PutUint32(data[i*size:], bits)
			case GGMLTypeBF16:
				binary.LittleEndian.PutUint16(data[i*size:], uint16(bits>>16))
			case GGMLTypeF16:
				binary.LittleEndian.PutUint16(data[i*size:], F32ToF16(value))
			}
		}
		return LittleBitScale{Raw: data, Type: typ}
	}
	factors := func(rank int) LittleBitFactors {
		return LittleBitFactors{
			Rank: rank, U: matrix(rows, rank), V: matrix(rank, cols),
			U1: scales(rows, GGMLTypeBF16), U2: scales(rank, GGMLTypeF16),
			V1: scales(rank, GGMLTypeF32), V2: scales(cols, GGMLTypeF16),
		}
	}
	w := Weight{Rows: rows, Cols: cols, Type: GGMLTypeI32, LittleBit: &LittleBitWeight{Main: factors(rank)}}
	if residualRank > 0 {
		residual := factors(residualRank)
		w.LittleBit.Residual = &residual
	}
	if bias {
		w.LittleBit.Bias = scales(rows, GGMLTypeBF16)
	}
	return w
}

func littleBitMetalTestInputs(cols, batch int) []float32 {
	random := rand.New(rand.NewSource(19))
	x := make([]float32, cols*batch)
	for i := range x {
		x[i] = float32(random.Intn(31)-15) / 16
	}
	return x
}

func TestLittleBitMetalParity(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	for _, c := range []struct {
		rows, cols, rank, residual, batch int
		bias                              bool
	}{
		{3, 35, 5, 0, 1, false},
		{67, 133, 33, 9, 7, true},
		{512, 1056, 257, 33, 2, true},
	} {
		label := fmt.Sprintf("%dx%d_rank%d_residual%d_batch%d", c.rows, c.cols, c.rank, c.residual, c.batch)
		t.Run(label, func(t *testing.T) {
			w := littleBitMetalTestWeight(c.rows, c.cols, c.rank, c.residual, c.bias)
			w.LittleBit.Metal = newMetalLittleBitWeight(w.LittleBit, w.Rows, w.Cols)
			if w.LittleBit.Metal == nil {
				t.Fatalf("upload LittleBit projection: %s", MetalError())
			}
			handle := w.LittleBit.Metal
			defer releaseMetalLittleBitWeight(handle)
			x := littleBitMetalTestInputs(w.Cols, c.batch)
			batch := make([]float32, c.batch*c.rows)
			if !matvecMetalLittleBitBatch(handle, x, batch, c.batch) {
				t.Fatalf("dispatch LittleBit batch: %s", MetalError())
			}
			var got, want []float32
			for token := range c.batch {
				input := x[token*c.cols : (token+1)*c.cols]
				w.MatvecInto(input, &got)
				w.LittleBit.matvecInto(input, w.Rows, w.Cols, &want, false)
				for i := range want {
					tolerance := 3e-4 + 3e-6*math.Abs(float64(want[i]))
					if !(math.Abs(float64(got[i]-want[i])) <= tolerance) ||
						!(math.Abs(float64(batch[token*c.rows+i]-want[i])) <= tolerance) {
						t.Fatalf("token %d output %d: single=%g batch=%g CPU=%g", token, i, got[i], batch[token*c.rows+i], want[i])
					}
				}
			}
			if weightBytes := metalbackend.LittleBitWeightBytes(handle.handle); weightBytes == 0 || weightBytes >= 4*c.rows*c.cols {
				t.Fatalf("unexpected Metal factor storage: %d bytes", weightBytes)
			}
			releaser := weightResourceReleaser{}
			releaser.release(&w)
			if w.LittleBit.Metal != nil || metalbackend.LittleBitWeightBytes(handle.handle) != 0 ||
				matvecMetalLittleBitBatch(handle, x, batch, c.batch) {
				t.Fatal("LittleBit Metal handle still live after releasing weights")
			}
		})
	}
	t.Run("F16 subnormal scale", func(t *testing.T) {
		one := LittleBitScale{Raw: []byte{0, 0x3c}, Type: GGMLTypeF16}
		subnormal := LittleBitScale{Raw: []byte{1, 0}, Type: GGMLTypeF16}
		w := &LittleBitWeight{Main: LittleBitFactors{
			Rank: 1, U: make([]byte, 4), V: make([]byte, 4),
			U1: one, U2: one, V1: one, V2: subnormal,
		}}
		handle := newMetalLittleBitWeight(w, 1, 1)
		defer releaseMetalLittleBitWeight(handle)
		x := []float32{1 << 24}
		out := []float32{0}
		if !matvecMetalLittleBitBatch(handle, x, out, 1) {
			t.Fatalf("dispatch LittleBit: %s", MetalError())
		}
		if out[0] != 1 {
			t.Fatalf("scaled subnormal should equal one, got %g", out[0])
		}
	})
}

func TestLittleBitMetalGGUFGenerate(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	for _, variant := range []uint32{1, 2} {
		t.Run(fmt.Sprintf("variant%d", variant), func(t *testing.T) {
			tied := variant == 2
			f := newLittleBitModelFixture(t, 40, 45, 2, 4, 2, 35, 7, variant, tied)
			data := writeLittleBitFixture(t, f.meta, f.packed)
			path := filepath.Join(t.TempDir(), "littlebit.gguf")
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true, LogWriter: io.Discard})
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			for _, l := range r.standard.Layers {
				for _, w := range [...]Weight{l.WQ, l.WK, l.WV, l.WO, l.W1, l.W2, l.W3} {
					if w.LittleBit == nil || w.LittleBit.Metal == nil || w.Metal != nil || w.F32 != nil {
						t.Fatalf("expected native Metal projection: %s", MetalError())
					}
				}
			}
			if !tied && (r.standard.Output.LittleBit == nil || r.standard.Output.LittleBit.Metal == nil) {
				t.Fatalf("expected native Metal output projection: %s", MetalError())
			}
			compareLittleBitModel(t, r, f.reference(t))
			factorBytes, scratch := r.LittleBitMetalMemory()
			t.Logf("loaded Metal factors %d B; shared projection scratch %d B", factorBytes, scratch)
			if factorBytes == 0 || scratch == 0 {
				t.Fatal("missing LittleBit Metal buffers")
			}
			if err := r.Close(); err != nil {
				t.Fatal(err)
			}
			if factors, _ := r.LittleBitMetalMemory(); factors != 0 {
				t.Fatalf("Metal factor buffers retained after Close: %d B", factors)
			}
		})
	}
}

func TestLittleBitMetalMixedFallback(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	f := newLittleBitModelFixture(t, 40, 45, 1, 4, 2, 35, 7, 2, true)
	keep := map[string]bool{
		"blk.0.attn_q": true, "blk.0.attn_v": true,
		"blk.0.ffn_gate": true, "blk.0.ffn_down": true,
	}
	var tensors []ggufTensor
	var params, payload uint64
	for _, tensor := range f.packed {
		if name, _, found := strings.Cut(tensor.name, littleBitTensorMarker); found && keep[name] {
			tensors = append(tensors, tensor)
			payload += uint64(len(tensor.data))
		}
	}
	for _, tensor := range f.dense {
		if keep[strings.TrimSuffix(tensor.name, ".weight")] {
			params += tensor.dims[0] * tensor.dims[1]
		} else {
			tensors = append(tensors, tensor)
		}
	}
	f.meta[littleBitNamespace+"original_parameters"] = MetaValue{Kind: "u64", Value: params}
	f.meta[littleBitNamespace+"effective_bpw"] = MetaValue{
		Kind: "f64", Value: 8 * float64(payload) / float64(params),
	}
	path := filepath.Join(t.TempDir(), "mixed.gguf")
	if err := os.WriteFile(path, writeLittleBitFixture(t, f.meta, tensors), 0600); err != nil {
		t.Fatal(err)
	}
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	layer := r.standard.Layers[0]
	if metalLittleBitProjection(layer.WQ) == nil || metalLittleBitProjection(layer.WV) == nil ||
		metalLittleBitProjection(layer.W1) == nil || metalLittleBitProjection(layer.W2) == nil ||
		layer.WK.LittleBit != nil || layer.W3.LittleBit != nil {
		t.Fatal("invalid mixed graph fixture")
	}
	// Test a group containing a GPU projection, a regular projection, and a
	// factorized CPU fallback as would happen when a single upload fails.
	releaseMetalLittleBitWeight(layer.WV.LittleBit.Metal)
	layer.WV.LittleBit.Metal = nil
	compareLittleBitModel(t, r, f.reference(t))
}

func BenchmarkLittleBitMetal(b *testing.B) {
	if !MetalAvailable() {
		b.Skip(MetalError())
	}
	const rows, cols = 4096, 4096
	for _, rank := range []int{32, 512} {
		for _, batch := range []int{1, 32} {
			label := fmt.Sprintf("rank%d/batch%d", rank, batch)
			b.Run(label, func(b *testing.B) {
				w := littleBitMetalTestWeight(rows, cols, rank, 0, false)
				x := littleBitMetalTestInputs(cols, batch)
				out := make([]float32, rows*batch)
				b.Run("CPU", func(b *testing.B) {
					ys := make([][]float32, batch)
					for token := range batch {
						ys[token] = out[token*rows : (token+1)*rows]
					}
					b.ReportAllocs()
					for b.Loop() {
						for token := range batch {
							w.MatvecInto(x[token*cols:(token+1)*cols], &ys[token])
						}
					}
					b.ReportMetric(float64(batch)*float64(b.N)/b.Elapsed().Seconds(), "vectors/s")
				})
				b.Run("Metal", func(b *testing.B) {
					handle := newMetalLittleBitWeight(w.LittleBit, w.Rows, w.Cols)
					if handle == nil {
						b.Fatalf("upload LittleBit projection: %s", MetalError())
					}
					defer releaseMetalLittleBitWeight(handle)
					if !matvecMetalLittleBitBatch(handle, x, out, batch) {
						b.Fatalf("warmup LittleBit: %s", MetalError())
					}
					var want []float32
					for token := range batch {
						w.LittleBit.matvecInto(x[token*cols:(token+1)*cols], rows, cols, &want, false)
						for i, value := range want {
							if match := out[token*rows+i]; !(math.Abs(float64(match-value)) <= 0.008+2e-5*math.Abs(float64(value))) {
								b.Fatalf("output %d: Metal=%g CPU=%g", i, match, value)
							}
						}
					}
					b.ReportAllocs()
					for b.Loop() {
						if !matvecMetalLittleBitBatch(handle, x, out, batch) {
							b.Fatalf("dispatch LittleBit: %s", MetalError())
						}
					}
					b.ReportMetric(float64(batch)*float64(b.N)/b.Elapsed().Seconds(), "vectors/s")
					b.ReportMetric(float64(metalbackend.LittleBitWeightBytes(handle.handle)), "GPU-weight-B")
					b.ReportMetric(float64(metalbackend.LittleBitScratchBytes()), "GPU-shared-scratch-B")
				})
			})
		}
	}
}

func uploadLittleBitGraphWeights(tb testing.TB, weights ...Weight) {
	tb.Helper()
	for _, w := range weights {
		h := newMetalLittleBitWeight(w.LittleBit, w.Rows, w.Cols)
		if h == nil {
			tb.Fatalf("upload scoped LittleBit projection: %s", MetalError())
		}
		tb.Cleanup(func() { releaseMetalLittleBitWeight(h) })
		w.LittleBit.Metal = h
	}
}

func TestLittleBitMetalGroupedGraph(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	const cols, hidden = 67, 73
	q := littleBitMetalTestWeight(19, cols, 35, 11, true)
	k := littleBitMetalTestWeight(17, cols, 17, 0, false)
	v := littleBitMetalTestWeight(13, cols, 3, 38, true)
	gate := littleBitMetalTestWeight(hidden, cols, 33, 27, true)
	up := littleBitMetalTestWeight(hidden, cols, 13, 0, false)
	down := littleBitMetalTestWeight(cols, hidden, 65, 9, true)
	uploadLittleBitGraphWeights(t, q, k, v, gate, up, down)
	for _, batch := range []int{1, 5} {
		t.Run(fmt.Sprintf("batch%d", batch), func(t *testing.T) {
			x := littleBitMetalTestInputs(cols, batch)
			qOut, kOut, vOut := make([]float32, batch*q.Rows), make([]float32, batch*k.Rows), make([]float32, batch*v.Rows)
			if matvecMetalLittleBitQKV(q, k, v, x, qOut, kOut[:len(kOut)-1], vOut, batch) ||
				matvecMetalLittleBitQKV(q, k, down, x, qOut, kOut, vOut, batch) {
				t.Fatal("grouped QKV accepted a short slab or mixed input width")
			}
			if !matvecMetalLittleBitQKV(q, k, v, x, qOut, kOut, vOut, batch) {
				t.Fatalf("grouped QKV: %s", MetalError())
			}
			for i, w := range []Weight{q, k, v} {
				want := make([]float32, batch*w.Rows)
				if !matvecMetalLittleBitBatch(w.LittleBit.Metal, x, want, batch) {
					t.Fatalf("separate projection: %s", MetalError())
				}
				got := [...][]float32{qOut, kOut, vOut}[i]
				for j, value := range got {
					if value != want[j] {
						t.Fatalf("projection %d[%d] = %g, separate Metal result %g", i, j, value, want[j])
					}
				}
			}
			var got []float32
			if matvecMetalLittleBitSwiGLUInto(gate, v, down, x, batch, &got) ||
				matvecMetalLittleBitSwiGLUInto(gate, up, q, x, batch, &got) {
				t.Fatal("SwiGLU accepted incompatible projections")
			}
			if !matvecMetalLittleBitSwiGLUInto(gate, up, down, x, batch, &got) {
				t.Fatalf("fused LittleBit SwiGLU: %s", MetalError())
			}
			g, u, h := make([]float32, batch*hidden), make([]float32, batch*hidden), make([]float32, batch*hidden)
			want := make([]float32, batch*cols)
			if !matvecMetalLittleBitBatch(gate.LittleBit.Metal, x, g, batch) ||
				!matvecMetalLittleBitBatch(up.LittleBit.Metal, x, u, batch) {
				t.Fatalf("separate gate/up: %s", MetalError())
			}
			for i := range g {
				h[i] = g[i] / (1 + float32(math.Exp(-float64(g[i])))) * u[i]
			}
			if !matvecMetalLittleBitBatch(down.LittleBit.Metal, h, want, batch) {
				t.Fatalf("separate down: %s", MetalError())
			}
			for i, value := range got {
				if !finite32(value) || math.Abs(float64(value-want[i])) > 0.001+2e-6*math.Abs(float64(want[i])) {
					t.Fatalf("fused SwiGLU output %d = %g, separate result %g", i, value, want[i])
				}
			}
		})
	}
}

func BenchmarkLittleBitMetalGroupedGraph(b *testing.B) {
	if !MetalAvailable() {
		b.Skip(MetalError())
	}
	// Actual Qwen3-8B LittleBit checkpoint widths and ranks, with synthetic
	// packed signs and scales to keep this benchmark independent of downloads.
	const dim, kv, hidden = 4096, 1024, 12288
	q := littleBitMetalTestWeight(dim, dim, 544, 544, false)
	k := littleBitMetalTestWeight(kv, dim, 208, 208, false)
	v := littleBitMetalTestWeight(kv, dim, 208, 208, false)
	gate := littleBitMetalTestWeight(hidden, dim, 824, 824, false)
	up := littleBitMetalTestWeight(hidden, dim, 824, 824, false)
	down := littleBitMetalTestWeight(dim, hidden, 824, 824, false)
	uploadLittleBitGraphWeights(b, q, k, v, gate, up, down)
	for _, batch := range []int{1, 26} {
		b.Run(fmt.Sprintf("batch%d", batch), func(b *testing.B) {
			x := littleBitMetalTestInputs(dim, batch)
			qOut, kOut, vOut := make([]float32, dim*batch), make([]float32, kv*batch), make([]float32, kv*batch)
			g, u, h := make([]float32, hidden*batch), make([]float32, hidden*batch), make([]float32, hidden*batch)
			out := make([]float32, dim*batch)
			for _, c := range []struct {
				name string
				run  func() bool
			}{
				{"QKV/separate", func() bool {
					return matvecMetalLittleBitBatch(q.LittleBit.Metal, x, qOut, batch) &&
						matvecMetalLittleBitBatch(k.LittleBit.Metal, x, kOut, batch) &&
						matvecMetalLittleBitBatch(v.LittleBit.Metal, x, vOut, batch)
				}},
				{"QKV/grouped", func() bool { return matvecMetalLittleBitQKV(q, k, v, x, qOut, kOut, vOut, batch) }},
				{"SwiGLU/separate", func() bool {
					if !matvecMetalLittleBitBatch(gate.LittleBit.Metal, x, g, batch) ||
						!matvecMetalLittleBitBatch(up.LittleBit.Metal, x, u, batch) {
						return false
					}
					siluMulF32(g, u, h)
					return matvecMetalLittleBitBatch(down.LittleBit.Metal, h, out, batch)
				}},
				{"SwiGLU/grouped", func() bool { return matvecMetalLittleBitSwiGLUInto(gate, up, down, x, batch, &out) }},
			} {
				b.Run(c.name, func(b *testing.B) {
					if !c.run() {
						b.Fatalf("LittleBit warmup: %s", MetalError())
					}
					b.ReportAllocs()
					for b.Loop() {
						if !c.run() {
							b.Fatalf("LittleBit dispatch: %s", MetalError())
						}
					}
				})
			}
		})
	}
}
