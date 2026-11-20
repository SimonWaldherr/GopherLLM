package engine

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// BenchmarkLittleBitModel compares execution on a complete synthetic decoder.
// The same dense oracle weights feed all conventional quantization fixtures.
func BenchmarkLittleBitModel(b *testing.B) {
	const promptTokens, decodeTokens = 32, 16
	fixture := newLittleBitModelFixture(b, 256, 768, 4, 8, 4, 8, 0, 2, true)
	original := uint64(0)
	for _, tensor := range fixture.dense {
		original += uint64(len(tensor.data) / 4)
	}
	for _, format := range []string{"LittleBit2", "Q1_0", "Q2_K", "Q4_K_M"} {
		b.Run(format, func(b *testing.B) {
			path := filepath.Join(b.TempDir(), "model.gguf")
			tensors := fixture.packed
			meta := fixture.meta
			if format != "LittleBit2" {
				meta = make(map[string]MetaValue)
				for k, v := range fixture.meta {
					if !strings.HasPrefix(k, littleBitNamespace) {
						meta[k] = v
					}
				}
				tensors = quantizedLittleBitBenchTensors(fixture.dense, format)
			}
			raw := writeLittleBitFixture(b, meta, tensors)
			if err := os.WriteFile(path, raw, 0600); err != nil {
				b.Fatal(err)
			}
			r, info, err := RunnerFromPath(path)
			if err != nil {
				b.Fatal(err)
			}
			defer r.Close()
			c := r.Config()
			k, v, head, kvHeads, value := r.cacheDims()
			cache := NewKVCache(c.NLayers, k, v, promptTokens+decodeTokens)
			buf := NewDecodeBuffer(c, head, kvHeads, value)
			prompt := make([]uint32, promptTokens)
			for i := range prompt {
				prompt[i] = 3 + uint32(i%10)
			}
			var logits []float32
			var prefill, decode time.Duration
			run := func() (time.Duration, time.Duration) {
				start := time.Now()
				ForwardBatchInto(c, r.standard, cache, buf, prompt, 0, true, &logits)
				middle := time.Now()
				token := argmaxFiniteToken(logits)
				for pos := promptTokens; pos < promptTokens+decodeTokens; pos++ {
					ForwardInto(c, r.standard, cache, buf, token, pos, &logits)
					token = argmaxFiniteToken(logits)
				}
				return middle.Sub(start), time.Since(middle)
			}
			run()
			b.ResetTimer()
			for range b.N {
				p, d := run()
				prefill += p
				decode += d
			}
			b.StopTimer()
			a := AnalyzeGGUF(r.GGUF(), nil)
			b.ReportMetric(float64(info.FileSizeBytes), "GGUF-B")
			b.ReportMetric(float64(a.FileBytes), "tensor-B")
			b.ReportMetric(8*float64(a.FileBytes)/float64(original), "tensor-bpw")
			b.ReportMetric(float64(promptTokens*b.N)/prefill.Seconds(), "prefill-tok/s")
			b.ReportMetric(float64(decodeTokens*b.N)/decode.Seconds(), "decode-tok/s")
			b.ReportMetric(float64(kvCacheBytes(c.NLayers, k, v, promptTokens+decodeTokens, kvF32)), "KV-B")
		})
	}
}

func quantizedLittleBitBenchTensors(dense []ggufTensor, format string) []ggufTensor {
	tensors := make([]ggufTensor, len(dense))
	hdr := &GGUFFile{}
	for _, t := range dense {
		hdr.Tensors = append(hdr.Tensors, TensorInfo{Name: t.name, DType: t.dtype, Dims: t.dims})
	}
	preset, _ := ParseCompressOptions(format)
	planner := newCompressionPlanner(hdr, preset)
	for i, t := range dense {
		tensors[i] = t
		if len(t.dims) != 2 || t.name == "token_embd.weight" {
			continue
		}
		typ, _ := planner.plan(hdr.Tensors[i])
		quantize := quantizeRowFor(typ)
		if format == "Q1_0" {
			typ, quantize = GGMLTypeQ1_0, quantizeLittleBitBenchQ1
		}
		if quantize == nil {
			continue
		}
		cols, rows := int(t.dims[0]), int(t.dims[1])
		values, _ := tensorToF32(t.dtype, t.data, cols*rows)
		size, _ := typ.DataSize(rows * cols)
		packed := make([]byte, 0, size)
		for j := range rows {
			packed = append(packed, quantize(values[j*cols:(j+1)*cols], cols)...)
		}
		tensors[i].dtype, tensors[i].data = typ, packed
	}
	return tensors
}

// The benchmark only needs a valid Q1_0 fixture. This sign-and-mean encoder
// deliberately stays here instead of defining a new compression policy.
func quantizeLittleBitBenchQ1(row []float32, cols int) []byte {
	data := make([]byte, cols/128*18)
	for b := range cols / 128 {
		chunk := data[b*18:][:18]
		var mean float64
		for j, value := range row[b*128:][:128] {
			mean += math.Abs(float64(value))
			if value >= 0 {
				chunk[2+j/8] |= 1 << (j % 8)
			}
		}
		binary.LittleEndian.PutUint16(chunk, F32ToF16(float32(mean/128)))
	}
	return data
}
