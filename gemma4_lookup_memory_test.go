package gopherllm

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"testing"
)

func TestNativeGemma4LookupStorage(t *testing.T) {
	for _, typ := range []GGMLType{GGMLTypeF32, GGMLTypeF16, GGMLTypeBF16} {
		for _, separate := range []bool{false, true} {
			for _, borrow := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/separate=%t/borrow=%t", typ, separate, borrow), func(t *testing.T) {
					data := buildTinyNativeGemma4E2BGGUF()
					g, err := ParseGGUFQuiet(data)
					if err != nil {
						t.Fatal(err)
					}
					var embedding TensorInfo
					for i := range g.Tensors {
						tensor := &g.Tensors[i]
						if tensor.Name != "token_embd.weight" && tensor.Name != "per_layer_token_embd.weight" {
							continue
						}
						raw := data[g.DataOffset+int(tensor.Offset):]
						for n := 0; n < tensor.Numel() && typ != GGMLTypeF32; n++ {
							bits := binary.LittleEndian.Uint32(raw[n*4:])
							half := uint16(bits >> 16)
							if typ == GGMLTypeF16 {
								half = F32ToF16(math.Float32frombits(bits))
							}
							binary.LittleEndian.PutUint16(raw[n*2:], half)
						}
						tensor.DType = typ
						if tensor.Name == "token_embd.weight" {
							embedding = *tensor
						}
					}
					if separate {
						embedding.Name = "output.weight"
						g.Tensors = append(g.Tensors, embedding)
					}
					cfg, weights, err := loadNativeGemma4Model(data, g, borrow, true, false, io.Discard)
					if err != nil {
						t.Fatal(err)
					}
					ple := weights.PerLayer.TokenEmbd
					if (ple.F32 == nil) != borrow || ple.Prepared != nil || ple.Metal != nil {
						t.Fatal("PLE lookup has unexpected storage or projection resources")
					}
					if (weights.TokenEmbd.F32 == nil) != (borrow && separate) || weights.Output.F32 == nil {
						t.Fatal("row-only input or tied output storage changed incorrectly")
					}
					oldPLE := *weights.PerLayer
					oldPLE.TokenEmbd, err = loadWeight(data, g.DataOffset, "per_layer_token_embd.weight", indexTensors(g), inferTensorSizes(data, g), false, false, true, false)
					if err != nil {
						t.Fatal(err)
					}
					reference := weights
					reference.PerLayer = &oldPLE
					reference.TokenEmbd, err = loadWeight(data, g.DataOffset, "token_embd.weight", indexTensors(g), inferTensorSizes(data, g), false, false, true, false)
					if err != nil {
						t.Fatal(err)
					}
					k1 := NewKVCache(cfg.NLayers, cfg.NKVHeads*cfg.HeadDim, cfg.NKVHeads*cfg.ValueDim, 8)
					k2 := NewKVCache(cfg.NLayers, cfg.NKVHeads*cfg.HeadDim, cfg.NKVHeads*cfg.ValueDim, 8)
					b1 := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
					b2 := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
					var want, got []float32
					for pos, token := range []uint32{1, 4, 7, 2} {
						ForwardGemma4Into(cfg, reference, k1, b1, token, pos, &want)
						ForwardGemma4Into(cfg, weights, k2, b2, token, pos, &got)
						requireMoEPrefillExact(t, got, want)
					}
				})
			}
		}
	}
}

func TestGemma4CorePrefaultKeepsLookupTableCold(t *testing.T) {
	data := buildTinyNativeGemma4E2BGGUF()
	g, err := ParseGGUFQuiet(data)
	if err != nil {
		t.Fatal(err)
	}
	ranges, _ := corePrefaultRanges(data, g)
	for _, tensor := range g.Tensors {
		size, ok := tensor.DType.DataSize(tensor.Numel())
		if !ok {
			t.Fatal("invalid fixture type")
		}
		start := g.DataOffset + int(tensor.Offset)
		for _, offset := range []int{start, start + size - 1} {
			warm := false
			for _, r := range ranges {
				warm = warm || (offset >= r.start && offset < r.end)
			}
			if want := tensor.Name != "per_layer_token_embd.weight"; warm != want {
				t.Fatalf("%s byte %d: warm=%t, want %t", tensor.Name, offset, warm, want)
			}
		}
	}
}
