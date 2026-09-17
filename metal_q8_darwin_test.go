//go:build darwin && cgo && metal

package gopherllm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

func metalQ8Fixture(t *testing.T, rows, cols int, seed int64) (Weight, *metalbackend.Weight) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	raw := make([]byte, 0, rows*(cols/32)*34)
	for range rows {
		raw = append(raw, randomQ8_0Row(rng, cols)...)
	}
	w := metalbackend.PrepareQ8_0(raw, rows, cols, false)
	if w == nil {
		t.Fatal(MetalError())
	}
	t.Cleanup(func() { metalbackend.Release(w) })
	return Weight{Raw: raw, Type: GGMLTypeQ8_0, Rows: rows, Cols: cols}, w
}

func assertMetalFiniteClose(t *testing.T, got, want []float32) {
	t.Helper()
	for i, v := range got {
		if !finite32(v) {
			t.Fatalf("nonfinite output %d: %g", i, v)
		}
	}
	assertMetalMatvecClose(t, got, want)
}

func TestMetalQ8MatvecAndArgmax(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	for _, cols := range []int{256, 2560, 9728} {
		t.Run(fmt.Sprint(cols), func(t *testing.T) {
			cpu, w := metalQ8Fixture(t, 67, cols, int64(cols))
			x := metalTestVector(cols)
			got := make([]float32, 67)
			if !metalbackend.MatvecQ8_0(w, x, got) {
				t.Fatal(MetalError())
			}
			assertMetalFiniteClose(t, got, cpu.Matvec(x))
			winner := argmaxFiniteToken(got)
			recent := []uint32{winner, winner, 1}
			penalized := append([]float32(nil), got...)
			applyRepeatPenalty(penalized, recent, 1.7)
			tok, ok := metalbackend.ArgmaxQ8_0Penalized(w, x, recent, 1.7)
			if !ok || tok != argmaxFiniteToken(penalized) {
				t.Fatalf("argmax %d/%v want %d", tok, ok, argmaxFiniteToken(penalized))
			}
			clear(x)
			tok, ok = metalbackend.ArgmaxQ8_0Penalized(w, x, nil, 1)
			if !ok || tok != 0 {
				t.Fatalf("tie argmax %d/%v", tok, ok)
			}
			if metalbackend.MatvecQ8_0(w, x[:cols-1], got) {
				t.Fatal("short input accepted")
			}
			if _, ok := metalbackend.ArgmaxQ8_0Penalized(w, x, nil, float32(math.NaN())); ok {
				t.Fatal("NaN penalty accepted")
			}
		})
	}
	if metalbackend.PrepareQ8_0(make([]byte, 33), 1, 32, false) != nil {
		t.Fatal("short weights accepted")
	}
	if metalbackend.PrepareQ8_0(make([]byte, 34), math.MaxInt, 256, false) != nil {
		t.Fatal("oversized shape accepted")
	}
	if prepareMetalWeight(nil, GGMLTypeQ8_0, 2560, 2560, false) != nil {
		t.Fatal("narrow attention weight prepared")
	}
}

func TestMetalQ8SwiGLUBatch(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	gate, g := metalQ8Fixture(t, 512, 256, 811)
	up, u := metalQ8Fixture(t, 512, 256, 812)
	down, d := metalQ8Fixture(t, 67, 512, 813)
	for _, batch := range []int{1, 2, 3, 4, 5, 8, 16, 31, 32, 35, 129, 256, 4} {
		t.Run(fmt.Sprint(batch), func(t *testing.T) {
			xs := metalTestVector(batch * 256)
			got := make([]float32, batch*67)
			want := make([]float32, len(got))
			for i := range xs {
				xs[i] *= 0.01
			}
			if !metalbackend.MatvecQ8_0SwiGLUBatch(g, u, d, xs, got, batch) {
				t.Fatal(MetalError())
			}
			for token := range batch {
				x := xs[token*256 : (token+1)*256]
				ga := gate.Matvec(x)
				uv := up.Matvec(x)
				for i := range ga {
					ga[i] = (ga[i] / (1 + float32(math.Exp(float64(-ga[i]))))) * uv[i]
				}
				copy(want[token*67:], down.Matvec(ga))
			}
			assertMetalFiniteClose(t, got, want)
		})
	}
	out := []float32{123}
	if metalbackend.MatvecQ8_0SwiGLUBatch(g, u, d, nil, out, 257) || out[0] != 123 {
		t.Fatal("invalid batch modified output")
	}
}

func TestMetalCoalescedPipelines(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	if !metalbackend.CoalescedAvailable() {
		t.Fatal("coalesced pipelines unavailable:", MetalError())
	}
}

func TestMetalQ8GreedyWeightBackend(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	w, handle := metalQ8Fixture(t, 8192, 256, 901)
	w.Metal = &MetalWeight{q8: handle, typ: GGMLTypeQ8_0, rows: w.Rows, cols: w.Cols}
	xs := [][]float32{metalTestVector(256), make([]float32, 256)}
	var got []uint32
	if !w.ArgmaxMatvecBatch(xs, &got) {
		t.Fatal("Q8 verification batch rejected")
	}
	for i, x := range xs {
		scores := w.Matvec(x)
		want := argmaxFiniteToken(scores)
		single, ok := w.ArgmaxMatvec(x)
		if !ok || single != want || got[i] != want {
			t.Fatalf("position%d single=%d batch=%d want=%d", i, single, got[i], want)
		}
	}
}

func TestMetalQ8ConcurrentDescriptor(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	_, handle := metalQ8Fixture(t, 67, 512, 902)
	const calls = 6
	xs := make([][]float32, calls)
	want := make([][]float32, calls)
	for i := range calls {
		xs[i] = metalTestVector(512)
		for j := range xs[i] {
			xs[i][j] *= float32(i+1) * 0.03
		}
		want[i] = make([]float32, 67)
		if !metalbackend.MatvecQ8_0(handle, xs[i], want[i]) {
			t.Fatal(MetalError())
		}
	}
	done := make(chan bool, calls)
	for i := range calls {
		go func() {
			out := make([]float32, 67)
			ok := metalbackend.MatvecQ8_0(handle, xs[i], out)
			for j := range out {
				ok = ok && math.Float32bits(out[j]) == math.Float32bits(want[i][j])
			}
			tok, valid := metalbackend.ArgmaxQ8_0Penalized(handle, xs[i], nil, 1)
			done <- ok && valid && tok == argmaxFiniteToken(want[i])
		}()
	}
	for range calls {
		if !<-done {
			t.Error("concurrent Metal descriptor produced different results")
		}
	}
}

func BenchmarkMetalQwenFFNBatch(b *testing.B) {
	if !MetalAvailable() {
		b.Skip(MetalError())
	}
	const inputCols, hiddenRows, outputRows = 2560, 9728, 2560
	rng := rand.New(rand.NewSource(198))
	gateRow := randomQ8_0Row(rng, inputCols)
	upRow := randomQ8_0Row(rng, inputCols)
	downRow := randomQ8_0Row(rng, hiddenRows)
	gateData := make([]byte, hiddenRows*len(gateRow))
	upData := make([]byte, hiddenRows*len(upRow))
	downData := make([]byte, outputRows*len(downRow))
	for r := range hiddenRows {
		copy(gateData[r*len(gateRow):], gateRow)
		copy(upData[r*len(upRow):], upRow)
	}
	for r := range outputRows {
		copy(downData[r*len(downRow):], downRow)
	}
	gateWeight := metalbackend.PrepareQ8_0(gateData, hiddenRows, inputCols, false)
	upWeight := metalbackend.PrepareQ8_0(upData, hiddenRows, inputCols, false)
	downWeight := metalbackend.PrepareQ8_0(downData, outputRows, hiddenRows, false)
	if gateWeight == nil || upWeight == nil || downWeight == nil {
		metalbackend.Release(gateWeight)
		metalbackend.Release(upWeight)
		metalbackend.Release(downWeight)
		b.Fatalf("prepare batched fused FFN Metal weights: %s", MetalError())
	}
	b.Cleanup(func() {
		metalbackend.Release(gateWeight)
		metalbackend.Release(upWeight)
		metalbackend.Release(downWeight)
	})
	gate := Weight{Raw: gateData, Type: GGMLTypeQ8_0, Rows: hiddenRows, Cols: inputCols}
	up := Weight{Raw: upData, Type: GGMLTypeQ8_0, Rows: hiddenRows, Cols: inputCols}
	down := Weight{Raw: downData, Type: GGMLTypeQ8_0, Rows: outputRows, Cols: hiddenRows}
	for _, batch := range []int{1, 32, 128} {
		b.Run(fmt.Sprintf("P%d", batch), func(b *testing.B) {
			xFlat := make([]float32, batch*inputCols)
			for i := range xFlat {
				xFlat[i] = float32((i*19)%59-29) / 17
			}
			xs := make([][]float32, batch)
			gateOut, upOut, hiddenOut, out := make([][]float32, batch), make([][]float32, batch), make([][]float32, batch), make([][]float32, batch)
			gateFlat, upFlat := make([]float32, batch*hiddenRows), make([]float32, batch*hiddenRows)
			hiddenFlat, outFlat := make([]float32, batch*hiddenRows), make([]float32, batch*outputRows)
			for token := range batch {
				xs[token] = xFlat[token*inputCols : (token+1)*inputCols]
				gateOut[token] = gateFlat[token*hiddenRows : (token+1)*hiddenRows]
				upOut[token] = upFlat[token*hiddenRows : (token+1)*hiddenRows]
				hiddenOut[token] = hiddenFlat[token*hiddenRows : (token+1)*hiddenRows]
				out[token] = outFlat[token*outputRows : (token+1)*outputRows]
			}
			b.Run("CPU_Q8_batch", func(b *testing.B) {
				withQ8Activations(true, func() {
					matvecBatch2(gate, up, xs, gateOut, upOut)
					for token := range batch {
						siluMulF32(gateOut[token], upOut[token], hiddenOut[token])
					}
					matvecBatch(down, hiddenOut, out)
					b.ReportAllocs()
					b.ResetTimer()
					for b.Loop() {
						matvecBatch2(gate, up, xs, gateOut, upOut)
						for token := range batch {
							siluMulF32(gateOut[token], upOut[token], hiddenOut[token])
						}
						matvecBatch(down, hiddenOut, out)
					}
				})
			})
			b.Run("Metal_resident", func(b *testing.B) {
				if !metalbackend.MatvecQ8_0SwiGLUBatch(gateWeight, upWeight, downWeight, xFlat, outFlat, batch) {
					b.Fatal(MetalError())
				}
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					if !metalbackend.MatvecQ8_0SwiGLUBatch(gateWeight, upWeight, downWeight, xFlat, outFlat, batch) {
						b.Fatal(MetalError())
					}
				}
			})
		})
	}
}

func BenchmarkMetalCoalescedFFNBatch(b *testing.B) {
	if !MetalAvailable() {
		b.Skip(MetalError())
	}
	const inputCols, hiddenRows, outputRows = 3072, 9216, 3072
	rng := rand.New(rand.NewSource(198))
	gateRow := randomQ4KRow(rng, inputCols)
	upRow := randomQ4KRow(rng, inputCols)
	downRow := randomQ6KRow(rng, hiddenRows)
	gateData := make([]byte, hiddenRows*len(gateRow))
	upData := make([]byte, hiddenRows*len(upRow))
	downData := make([]byte, outputRows*len(downRow))
	for r := range hiddenRows {
		copy(gateData[r*len(gateRow):], gateRow)
		copy(upData[r*len(upRow):], upRow)
	}
	for r := range outputRows {
		copy(downData[r*len(downRow):], downRow)
	}
	gateWeight := metalbackend.PrepareQ4K(gateData, hiddenRows, inputCols, false)
	upWeight := metalbackend.PrepareQ4K(upData, hiddenRows, inputCols, false)
	downWeight := metalbackend.PrepareQ6K(downData, outputRows, hiddenRows, false)
	if gateWeight == nil || upWeight == nil || downWeight == nil {
		metalbackend.Release(gateWeight)
		metalbackend.Release(upWeight)
		metalbackend.Release(downWeight)
		b.Fatalf("prepare batched fused FFN Metal weights: %s", MetalError())
	}
	b.Cleanup(func() {
		metalbackend.Release(gateWeight)
		metalbackend.Release(upWeight)
		metalbackend.Release(downWeight)
	})
	gate := Weight{Raw: gateData, Type: GGMLTypeQ4_K, Rows: hiddenRows, Cols: inputCols}
	up := Weight{Raw: upData, Type: GGMLTypeQ4_K, Rows: hiddenRows, Cols: inputCols}
	down := Weight{Raw: downData, Type: GGMLTypeQ6_K, Rows: outputRows, Cols: hiddenRows}
	for _, batch := range []int{1, 32} {
		b.Run(fmt.Sprintf("P%d", batch), func(b *testing.B) {
			xFlat := make([]float32, batch*inputCols)
			for i := range xFlat {
				xFlat[i] = float32((i*19)%59-29) / 17
			}
			xs := make([][]float32, batch)
			gateOut, upOut, hiddenOut, out := make([][]float32, batch), make([][]float32, batch), make([][]float32, batch), make([][]float32, batch)
			gateFlat, upFlat := make([]float32, batch*hiddenRows), make([]float32, batch*hiddenRows)
			hiddenFlat, outFlat := make([]float32, batch*hiddenRows), make([]float32, batch*outputRows)
			for token := range batch {
				xs[token] = xFlat[token*inputCols : (token+1)*inputCols]
				gateOut[token] = gateFlat[token*hiddenRows : (token+1)*hiddenRows]
				upOut[token] = upFlat[token*hiddenRows : (token+1)*hiddenRows]
				hiddenOut[token] = hiddenFlat[token*hiddenRows : (token+1)*hiddenRows]
				out[token] = outFlat[token*outputRows : (token+1)*outputRows]
			}
			b.Run("CPU_Q8_batch", func(b *testing.B) {
				withQ8Activations(true, func() {
					matvecBatch2(gate, up, xs, gateOut, upOut)
					for token := range batch {
						siluMulF32(gateOut[token], upOut[token], hiddenOut[token])
					}
					matvecBatch(down, hiddenOut, out)
					b.ReportAllocs()
					b.ResetTimer()
					for b.Loop() {
						matvecBatch2(gate, up, xs, gateOut, upOut)
						for token := range batch {
							siluMulF32(gateOut[token], upOut[token], hiddenOut[token])
						}
						matvecBatch(down, hiddenOut, out)
					}
				})
			})
			b.Run("Metal_resident", func(b *testing.B) {
				if !metalbackend.MatvecQ4K2SwiGLUQ6KBatch(gateWeight, upWeight, downWeight, xFlat, outFlat, batch) {
					b.Fatal(MetalError())
				}
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					if !metalbackend.MatvecQ4K2SwiGLUQ6KBatch(gateWeight, upWeight, downWeight, xFlat, outFlat, batch) {
						b.Fatal(MetalError())
					}
				}
			})
		})
	}
}

func TestMetalDecodeRetainsOriginalKernel(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	if !metalbackend.CoalescedAvailable() {
		t.Fatal(MetalError())
	}
	for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K} {
		for _, cols := range []int{512, 3072, 9216} {
			t.Run(fmt.Sprintf("%s/%d", typ, cols), func(t *testing.T) {
				const rows = 67
				rng := rand.New(rand.NewSource(int64(cols)))
				var raw []byte
				for range rows {
					if typ == GGMLTypeQ4_K {
						raw = append(raw, randomQ4KRow(rng, cols)...)
					} else {
						raw = append(raw, randomQ6KRow(rng, cols)...)
					}
				}
				var w *metalbackend.Weight
				var run func(*metalbackend.Weight, []float32, []float32) bool
				if typ == GGMLTypeQ4_K {
					w = metalbackend.PrepareQ4K(raw, rows, cols, false)
					run = metalbackend.MatvecQ4K
				} else {
					w = metalbackend.PrepareQ6K(raw, rows, cols, false)
					run = metalbackend.MatvecQ6K
				}
				if w == nil {
					t.Fatal(MetalError())
				}
				t.Cleanup(func() { metalbackend.Release(w) })
				x := metalTestVector(cols)
				want := make([]float32, rows)
				got := make([]float32, rows)
				t.Setenv("GOPHERLLM_METAL_COALESCED", "0")
				if !run(w, x, want) {
					t.Fatal(MetalError())
				}
				t.Setenv("GOPHERLLM_METAL_COALESCED", "1")
				if !run(w, x, got) {
					t.Fatal(MetalError())
				}
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("decode row%d changed", i)
					}
				}
			})
		}
	}
}

func TestMetalMatrixPipelines(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	if !metalbackend.MatrixAvailable() {
		t.Fatal(MetalError())
	}
}
func TestMetalMatrixQ4Q6Tails(t *testing.T)   { testMetalMatrixKTails(t, false) }
func TestMetalMatrixQ4DownTails(t *testing.T) { testMetalMatrixKTails(t, true) }
func testMetalMatrixKTails(t *testing.T, downQ4 bool) {
	t.Helper()
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	const inputCols, hiddenRows, outputRows = 512, 1024, 67
	randomDown := randomQ6KRow
	matvecDown := MatvecQ6KInto
	prepareDown := metalbackend.PrepareQ6K
	fused := metalbackend.MatvecQ4K2SwiGLUQ6KBatch
	if downQ4 {
		randomDown = randomQ4KRow
		matvecDown = MatvecQ4KInto
		prepareDown = metalbackend.PrepareQ4K
		fused = metalbackend.MatvecQ4K2SwiGLUQ4KBatch
	}
	for _, batch := range []int{1, 2, 16, 31, 32, 35, 129, 256} {
		rng := rand.New(rand.NewSource(197))
		gateData := make([]byte, 0, hiddenRows*144)
		upData := make([]byte, 0, hiddenRows*144)
		for range hiddenRows {
			gateData = append(gateData, randomQ4KRow(rng, inputCols)...)
			upData = append(upData, randomQ4KRow(rng, inputCols)...)
		}
		downData := make([]byte, 0, outputRows*(hiddenRows/256)*210)
		for range outputRows {
			downData = append(downData, randomDown(rng, hiddenRows)...)
		}
		inputs := make([]float32, batch*inputCols)
		for i := range inputs {
			inputs[i] = float32((i*17)%43-21) / 13
		}
		want := make([]float32, batch*outputRows)
		for token := 0; token < batch; token++ {
			x := inputs[token*inputCols : (token+1)*inputCols]
			gate, up := []float32{}, []float32{}
			hidden := make([]float32, hiddenRows)
			MatvecQ4KInto(gateData, x, hiddenRows, inputCols, &gate)
			MatvecQ4KInto(upData, x, hiddenRows, inputCols, &up)
			siluMulF32(gate, up, hidden)
			got := want[token*outputRows : (token+1)*outputRows]
			matvecDown(downData, hidden, outputRows, hiddenRows, &got)
		}
		gateWeight := metalbackend.PrepareQ4K(gateData, hiddenRows, inputCols, false)
		upWeight := metalbackend.PrepareQ4K(upData, hiddenRows, inputCols, false)
		downWeight := prepareDown(downData, outputRows, hiddenRows, false)
		if gateWeight == nil || upWeight == nil || downWeight == nil {
			metalbackend.Release(gateWeight)
			metalbackend.Release(upWeight)
			metalbackend.Release(downWeight)
			t.Fatalf("prepare batched fused FFN Metal weights: %s", MetalError())
		}
		defer metalbackend.Release(gateWeight)
		defer metalbackend.Release(upWeight)
		defer metalbackend.Release(downWeight)
		got := make([]float32, batch*outputRows)
		if !fused(gateWeight, upWeight, downWeight, inputs, got, batch) {
			t.Fatalf("batched fused FFN Metal matvec: %s", MetalError())
		}
		// Three matrix products and SiLU amplify rounding near cancellation.
		// Check a normwise error bound as well as every individual output.
		var energy, errorEnergy float64
		for i, v := range got {
			if !finite32(v) {
				t.Fatalf("nonfinite matrix output %d", i)
			}
			d := float64(v) - float64(want[i])
			energy += float64(want[i]) * float64(want[i])
			errorEnergy += d * d
		}
		relative := math.Sqrt(errorEnergy / math.Max(energy, 1))
		t.Logf("batch %d relative L2 error %.3g", batch, relative)
		if relative > 1e-5 {
			t.Fatalf("matrix relative L2 error %g", relative)
		}
		rms := math.Sqrt(energy / float64(len(want)))
		for i, v := range got {
			tolerance := 1e-5*math.Max(1, rms) + 1e-4*math.Abs(float64(want[i]))
			if math.Abs(float64(v)-float64(want[i])) > tolerance {
				t.Fatalf("output %d: %g want %g tolerance %g", i, v, want[i], tolerance)
			}
		}
	}
}
