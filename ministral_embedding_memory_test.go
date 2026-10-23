package gopherllm

import (
	"bytes"
	"io"
	"math"
	"testing"
)

func ministralEmbeddingMemoryFixture(t testing.TB, typ GGMLType, tied bool) ([]byte, *GGUFFile) {
	t.Helper()
	data := buildTinyQuantizedMistralGGUF()
	g, err := ParseGGUFQuiet(data)
	if err != nil {
		t.Fatal(err)
	}
	for i := range g.Tensors {
		info := &g.Tensors[i]
		if info.Name == "token_embd.weight" && typ == GGMLTypeQ4_K {
			info.DType = typ
			for row := 0; row < int(info.Dims[1]); row++ {
				packed := QuantizeRowQ4K(smallWeights(int(info.Dims[0]), 1+row), int(info.Dims[0]))
				copy(data[g.DataOffset+int(info.Offset)+row*len(packed):], packed)
			}
		}
	}
	if tied {
		for i := range g.Tensors {
			if g.Tensors[i].Name == "output.weight" {
				g.Tensors = append(g.Tensors[:i], g.Tensors[i+1:]...)
				break
			}
		}
	}
	return data, g
}

func TestMinistralEmbeddingPreparationKeepsTiedOutput(t *testing.T) {
	for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K} {
		for _, tied := range []bool{false, true} {
			for _, borrow := range []bool{false, true} {
				name := typ.String()
				if tied {
					name += "/tied"
				} else {
					name += "/untied"
				}
				if borrow {
					name += "/borrowed"
				} else {
					name += "/owned"
				}
				t.Run(name, func(t *testing.T) {
					data, g := ministralEmbeddingMemoryFixture(t, typ, tied)
					cfg, w, err := LoadModel(data, g, borrow, true, false, io.Discard)
					if err != nil {
						t.Fatal(err)
					}
					if got := w.TokenEmbd.Prepared != nil; got != tied {
						t.Fatalf("embedding prepared=%t; tied=%t", got, tied)
					}
					if w.Output.Prepared == nil {
						t.Fatal("vocabulary projection lost prepared scales")
					}
					// Compare against the old loader, which always prepared the
					// embedding. All row values and complete decode results must
					// remain exact despite removing those unused resources.
					embedding, err := loadWeight(data, g.DataOffset, "token_embd.weight", indexTensors(g), inferTensorSizes(data, g), false, borrow, true, false)
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(w.TokenEmbd.Raw, embedding.Raw) {
						t.Fatal("embedding packed rows changed")
					}
					before := w
					before.TokenEmbd = embedding
					kDim := cfg.NKVHeads * cfg.HeadDim
					newCache := func() *KVCache { return NewKVCache(cfg.NLayers, kDim, cfg.KVDim, 8) }
					refCache, gotCache := newCache(), newCache()
					refBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
					gotBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
					for pos, token := range []uint32{1, 4, 7, 9, 4, 3, 12, 2} {
						ref := Forward(cfg, before, refCache, refBuf, token, pos)
						got := Forward(cfg, w, gotCache, gotBuf, token, pos)
						for i := range ref {
							if math.Float32bits(ref[i]) != math.Float32bits(got[i]) {
								t.Fatalf("position %d logit %d differs: %v vs %v", pos, i, ref[i], got[i])
							}
						}
					}
					for layer := range refCache.K {
						for i := range refCache.K[layer] {
							if math.Float32bits(refCache.K[layer][i]) != math.Float32bits(gotCache.K[layer][i]) ||
								math.Float32bits(refCache.V[layer][i]) != math.Float32bits(gotCache.V[layer][i]) {
								t.Fatalf("layer %d KV row element %d differs", layer, i)
							}
						}
					}
				})
			}
		}
	}
}
