package engine

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type littleBitModelFixture struct {
	meta   map[string]MetaValue
	packed []ggufTensor
	dense  []ggufTensor
}

func newLittleBitModelFixture(tb testing.TB, dim, hidden, layers, heads, kvHeads, rank, residualRank int, variant uint32, tied bool) littleBitModelFixture {
	tb.Helper()
	base, err := ParseGGUFQuiet(buildTinyLlamaGGUF())
	if err != nil {
		tb.Fatal(err)
	}
	f := littleBitModelFixture{meta: base.Metadata}
	const vocab = 16
	headDim := dim / heads
	for key, v := range map[string]int{
		"embedding_length": dim, "feed_forward_length": hidden, "block_count": layers,
		"attention.head_count": heads, "attention.head_count_kv": kvHeads, "attention.key_length": headDim,
		"attention.value_length": headDim, "rope.dimension_count": headDim, "context_length": 128,
	} {
		f.meta["llama."+key] = MetaValue{Kind: "u32", Value: uint32(v)}
	}
	var originals, payload uint64
	rng := rand.New(rand.NewPCG(22, 17))
	common := func(name string, dims []uint64, values []float32) {
		tensor := ggufTensor{name: name, dims: dims, dtype: GGMLTypeF32, data: f32Bytes(values)}
		f.packed = append(f.packed, tensor)
		f.dense = append(f.dense, tensor)
	}
	projection := func(name string, rows, cols int, biasName string) {
		originals += uint64(rows) * uint64(cols)
		first := len(f.packed)
		dense := make([]float32, rows*cols)
		path := func(prefix string, rank int) {
			add := func(suffix string, dims []uint64, dt GGMLType, data []byte) {
				f.packed = append(f.packed, ggufTensor{name: prefix + suffix, dims: dims, dtype: dt, data: data})
			}
			// Keep unpacked signs here to build a separate dense reference.
			signs := func(suffix string, rows, cols int) []float32 {
				words := (cols + 31) / 32
				matrix := make([]float32, rows*cols)
				bytes := make([]byte, rows*words*4)
				for row := range rows {
					for col := range cols {
						sign := float32(1)
						if rng.Uint64()&1 != 0 {
							sign = -1
							bytes[row*words*4+col/8] |= 1 << (col % 8)
						}
						matrix[row*cols+col] = sign
					}
				}
				add(suffix, []uint64{uint64(words), uint64(rows)}, GGMLTypeI32, bytes)
				return matrix
			}
			scale := func(suffix string, n int, base float32) []float32 {
				values := make([]float32, n)
				bytes := make([]byte, n*2)
				for i := range values {
					values[i] = base * float32(3+rng.IntN(4)) / 4
					binary.LittleEndian.PutUint16(bytes[2*i:], F32ToF16(values[i]))
					values[i] = F16ToF32(binary.LittleEndian.Uint16(bytes[2*i:]))
				}
				add(suffix, []uint64{uint64(n)}, GGMLTypeF16, bytes)
				return values
			}
			u, v := signs("u", rows, rank), signs("v", rank, cols)
			u1, u2 := scale("u1", rows, 0.125), scale("u2", rank, 0.25)
			v1, v2 := scale("v1", rank, 0.25), scale("v2", cols, 0.25)
			for i := range rows {
				for j := range cols {
					var cell float64
					for k := range rank {
						cell += float64(u[i*rank+k]) * float64(u2[k]) * float64(v1[k]) * float64(v[k*cols+j])
					}
					dense[i*cols+j] += float32(float64(u1[i]) * cell * float64(v2[j]))
				}
			}
		}
		path(name+".lb.", rank)
		if residualRank > 0 {
			path(name+".lb.residual.", residualRank)
		}
		if biasName != "" {
			bias := f32Bytes(smallWeights(rows, 7))
			f.packed = append(f.packed, ggufTensor{name: name + ".lb.bias", dims: []uint64{uint64(rows)}, dtype: GGMLTypeF32, data: bias})
			f.dense = append(f.dense, ggufTensor{name: biasName, dims: []uint64{uint64(rows)}, dtype: GGMLTypeF32, data: bias})
		}
		for _, t := range f.packed[first:] {
			payload += uint64(len(t.data))
		}
		f.dense = append(f.dense, ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(dense)})
	}

	common("token_embd.weight", []uint64{uint64(dim), vocab}, smallWeights(vocab*dim, 1))
	common("output_norm.weight", []uint64{uint64(dim)}, onesF32(dim))
	for layer := range layers {
		prefix := fmt.Sprintf("blk.%d.", layer)
		common(prefix+"attn_norm.weight", []uint64{uint64(dim)}, onesF32(dim))
		common(prefix+"ffn_norm.weight", []uint64{uint64(dim)}, onesF32(dim))
		projection(prefix+"attn_q.weight", dim, dim, "")
		projection(prefix+"attn_k.weight", headDim*kvHeads, dim, "")
		projection(prefix+"attn_v.weight", headDim*kvHeads, dim, "")
		projection(prefix+"attn_output.weight", dim, dim, prefix+"attn_output.bias")
		projection(prefix+"ffn_gate.weight", hidden, dim, "")
		projection(prefix+"ffn_up.weight", hidden, dim, "")
		projection(prefix+"ffn_down.weight", dim, hidden, "")
	}
	if !tied {
		projection("output.weight", vocab, dim, "output.bias")
	}
	f.meta[littleBitNamespace+"version"] = MetaValue{Kind: "u32", Value: uint32(1)}
	f.meta[littleBitNamespace+"variant"] = MetaValue{Kind: "u32", Value: variant}
	f.meta[littleBitNamespace+"original_parameters"] = MetaValue{Kind: "u64", Value: originals}
	f.meta[littleBitNamespace+"effective_bpw"] = MetaValue{Kind: "f64", Value: 8 * float64(payload) / float64(originals)}
	return f
}

func writeLittleBitFixture(tb testing.TB, meta map[string]MetaValue, tensors []ggufTensor) []byte {
	tb.Helper()
	plan := make([]PlannedTensor, len(tensors))
	for i, tensor := range tensors {
		plan[i] = PlannedTensor{Name: tensor.name, Dims: tensor.dims, DType: tensor.dtype}
	}
	var out bytes.Buffer
	w, err := NewGGUFWriter(&out, meta, plan, 32)
	if err != nil {
		tb.Fatal(err)
	}
	for _, tensor := range tensors {
		if err := w.WriteTensor(tensor.data); err != nil {
			tb.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		tb.Fatal(err)
	}
	return out.Bytes()
}

func (f littleBitModelFixture) reference(tb testing.TB) *Runner {
	tb.Helper()
	meta := maps.Clone(f.meta)
	for k := range meta {
		if strings.HasPrefix(k, littleBitNamespace) {
			delete(meta, k)
		}
	}
	r, err := RunnerFromGGUFBytes(writeLittleBitFixture(tb, meta, f.dense))
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { r.Close() })
	return r
}

func compareLittleBitModel(t *testing.T, got, reference *Runner) {
	t.Helper()
	tokens := []uint32{1, 3, 4, 8, 11, 14, 5}
	c := got.Config()
	newState := func() (*KVCache, *DecodeBuffer) {
		k, v, head, kvHeads, value := got.cacheDims()
		return NewKVCache(c.NLayers, k, v, 128), NewDecodeBuffer(c, head, kvHeads, value)
	}
	rCache, rBuf := newState()
	gCache, gBuf := newState()
	var ref, pred []float32
	check := func(label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s, length %d != %d", label, len(got), len(want))
		}
		for i := range got {
			if !finite32(got[i]) || math.Abs(float64(got[i]-want[i])) > 3e-5*math.Max(1, math.Abs(float64(want[i]))) {
				t.Fatalf("%s[%d] = %g, want %g", label, i, got[i], want[i])
			}
		}
	}
	for i, token := range tokens {
		ForwardInto(c, reference.standard, rCache, rBuf, token, i, &ref)
		ForwardInto(c, got.standard, gCache, gBuf, token, i, &pred)
		check(fmt.Sprintf("logits at pos %d", i), pred, ref)
		if gotToken, ok := ArgmaxOutputToken(c, got.standard, gBuf); !ok || gotToken != uint32(slices.Index(ref, slices.Max(ref))) {
			t.Fatalf("greedy token %d (ok=%t), reference logits %v", gotToken, ok, ref)
		}
	}
	bCache, bBuf := newState()
	ForwardBatchInto(c, got.standard, bCache, bBuf, tokens, 0, true, &pred)
	check("batch logits", pred, ref)
	check("batch K", bCache.K[0], rCache.K[0])
	check("batch V", bCache.V[0], rCache.V[0])

	opts := DefaultGenerationOptions()
	opts.MaxTokens = 5
	opts.SystemPrompt = ""
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	a, err := got.Generate("a b c", opts)
	if err != nil {
		t.Fatal(err)
	}
	b, err := reference.Generate("a b c", opts)
	if err != nil {
		t.Fatal(err)
	}
	if a.Stats.GeneratedTokens == 0 || a.Stats.GeneratedTokens != b.Stats.GeneratedTokens || a.Text != b.Text {
		t.Fatalf("generated %d tokens %q, want %d %q", a.Stats.GeneratedTokens, a.Text, b.Stats.GeneratedTokens, b.Text)
	}
}

func TestLittleBitGGUFRoundTripAndGenerate(t *testing.T) {
	for _, tc := range []struct {
		name    string
		variant uint32
		tied    bool
	}{{"LittleBit1SeparateOutput", 1, false}, {"LittleBit2Tied", 2, true}} {
		t.Run(tc.name, func(t *testing.T) {
			f := newLittleBitModelFixture(t, 40, 45, 2, 4, 2, 35, 7, tc.variant, tc.tied)
			raw := writeLittleBitFixture(t, f.meta, f.packed)
			r, err := RunnerFromGGUFBytes(raw)
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			q := r.standard.Layers[0].WQ
			if q.F32 != nil || q.Raw != nil || q.LittleBit == nil || q.LittleBit.Residual == nil ||
				q.LittleBit.Main.Rank != 35 || q.LittleBit.Residual.Rank != 7 {
				t.Fatal("expected packed main and residual factors only")
			}
			info := indexTensors(r.GGUF())["blk.0.attn_q.weight.lb.u"]
			off := r.GGUF().DataOffset + int(info.Offset)
			if &q.LittleBit.Main.U[0] == &raw[off] {
				t.Fatal("in-memory runner borrows caller-owned GGUF bytes")
			}
			if (r.standard.Output.LittleBit == nil) != tc.tied {
				t.Fatal("wrong output weight layout")
			}
			a := AnalyzeGGUF(r.GGUF(), nil)
			var wantParams int64
			for _, d := range f.dense {
				// The composite bias is counted in payload, not original matrices.
				if !strings.HasSuffix(d.name, ".bias") {
					wantParams += int64(len(d.data) / 4)
				}
			}
			if a.Params != wantParams || a.LittleBitVariant != tc.variant || a.LittleBitBytes <= 0 || a.FileBytes <= a.LittleBitBytes {
				t.Fatalf("wrong LittleBit analysis: %+v", a)
			}
			compareLittleBitModel(t, r, f.reference(t))
			path := filepath.Join(t.TempDir(), "littlebit.gguf")
			if err := os.WriteFile(path, raw, 0600); err != nil {
				t.Fatal(err)
			}
			mapped, _, err := RunnerFromPath(path)
			if err != nil {
				t.Fatal(err)
			}
			defer mapped.Close()
			factors := mapped.standard.Layers[0].WQ.LittleBit.Main
			if &factors.U[0] != &mapped.mappedFile.Bytes()[off] {
				t.Fatal("packed U is not a borrowed file slice")
			}
			compareLittleBitModel(t, mapped, r)
		})
	}
}

func TestLittleBitDenseQwen3WithMixedProjections(t *testing.T) {
	f := newLittleBitModelFixture(t, 40, 45, 1, 4, 2, 9, 0, 2, false)
	f.meta["general.architecture"] = MetaValue{Kind: "str", Value: "qwen3"}
	for key, value := range maps.Clone(f.meta) {
		if name, ok := strings.CutPrefix(key, "llama."); ok {
			delete(f.meta, key)
			f.meta["qwen3."+name] = value
		}
	}
	for _, name := range []string{"blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight"} {
		values := smallWeights(10, 4)
		for i := range values {
			values[i] += 1
		}
		norm := ggufTensor{name: name, dims: []uint64{10}, dtype: GGMLTypeF32, data: f32Bytes(values)}
		f.packed = append(f.packed, norm)
		f.dense = append(f.dense, norm)
	}
	// Keep only Q factorized to exercise the ordinary projections beside it.
	var packed []ggufTensor
	var payload uint64
	for _, tensor := range f.packed {
		if strings.HasPrefix(tensor.name, "blk.0.attn_q.weight.lb.") {
			packed = append(packed, tensor)
			payload += uint64(len(tensor.data))
		}
	}
	for _, tensor := range f.dense {
		if tensor.name != "blk.0.attn_q.weight" {
			packed = append(packed, tensor)
		}
	}
	f.meta[littleBitNamespace+"original_parameters"] = MetaValue{Kind: "u64", Value: uint64(1600)}
	f.meta[littleBitNamespace+"effective_bpw"] = MetaValue{Kind: "f64", Value: 8 * float64(payload) / 1600}
	r, err := RunnerFromGGUFBytes(writeLittleBitFixture(t, f.meta, packed))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	compareLittleBitModel(t, r, f.reference(t))
}

func TestLittleBitSplitMmap(t *testing.T) {
	f := newLittleBitModelFixture(t, 40, 45, 1, 4, 2, 9, 6, 2, false)
	prefix := filepath.Join(t.TempDir(), "littlebit")
	var partitions [2][]ggufTensor
	for i, tensor := range f.packed {
		partitions[i%2] = append(partitions[i%2], tensor)
	}
	for i, tensors := range partitions {
		meta := make(map[string]MetaValue)
		if i == 0 {
			maps.Copy(meta, f.meta)
		}
		meta["split.no"] = MetaValue{Kind: "u16", Value: uint16(i)}
		meta["split.count"] = MetaValue{Kind: "u16", Value: uint16(2)}
		meta["split.tensors.count"] = MetaValue{Kind: "i32", Value: int32(len(f.packed))}
		if err := os.WriteFile(splitShardPath(prefix, i+1, 2), writeLittleBitFixture(t, meta, tensors), 0600); err != nil {
			t.Fatal(err)
		}
	}
	for _, opts := range []LoadOptions{{}, {OutOfCore: true}} {
		r, _, err := RunnerFromPathWithOptions(splitShardPath(prefix, 2, 2), opts)
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()
		info := indexTensors(r.GGUF())["blk.0.attn_q.weight.lb.u"]
		if len(r.extraMappedFiles) != 2 || len(info.Shard) == 0 ||
			&r.standard.Layers[0].WQ.LittleBit.Main.U[0] != &info.Shard[info.Offset] {
			t.Fatal("split factor is not a borrowed shard slice")
		}
		compareLittleBitModel(t, r, f.reference(t))
		if err := r.Close(); err != nil {
			t.Fatal(err)
		}
	}
	badMeta := map[string]MetaValue{
		"split.no":            {Kind: "u16", Value: uint16(1)},
		"split.count":         {Kind: "u16", Value: uint16(2)},
		"split.tensors.count": {Kind: "i32", Value: int32(len(f.packed) - 1)},
	}
	if err := os.WriteFile(splitShardPath(prefix, 2, 2), writeLittleBitFixture(t, badMeta, partitions[1]), 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := RunnerFromPath(splitShardPath(prefix, 1, 2)); err == nil || !strings.Contains(err.Error(), "split.tensors.count") {
		t.Fatalf("got %v for conflicting split.tensors.count", err)
	}
}

func TestLittleBitRejectsInvalidGGUF(t *testing.T) {
	fixture := newLittleBitModelFixture(t, 40, 45, 1, 4, 2, 35, 7, 1, true)
	const prefix = "blk.0.attn_q.weight.lb."
	type mutate func(*GGUFFile, []byte, map[string]int)
	tests := []struct {
		name string
		want string
		edit mutate
	}{
		{"wrong architecture", "architecture", func(g *GGUFFile, _ []byte, _ map[string]int) {
			g.Metadata["general.architecture"] = MetaValue{Kind: "str", Value: "qwen3moe"}
		}},
		{"missing version", "version", func(g *GGUFFile, _ []byte, _ map[string]int) {
			delete(g.Metadata, littleBitNamespace+"version")
		}},
		{"variant", "variant", func(g *GGUFFile, _ []byte, _ map[string]int) {
			g.Metadata[littleBitNamespace+"variant"] = MetaValue{Kind: "u32", Value: uint32(9)}
		}},
		{"missing U", "missing tensor", func(g *GGUFFile, _ []byte, idx map[string]int) {
			i := idx[prefix+"u"]
			g.Tensors = slices.Delete(g.Tensors, i, i+1)
		}},
		{"missing residual V2", "missing tensor", func(g *GGUFFile, _ []byte, idx map[string]int) {
			i := idx[prefix+"residual.v2"]
			g.Tensors = slices.Delete(g.Tensors, i, i+1)
		}},
		{"wrong I32 shape", "want I32", func(g *GGUFFile, _ []byte, idx map[string]int) {
			g.Tensors[idx[prefix+"u"]].Dims[0]++
		}},
		{"wrong I32 type", "want I32", func(g *GGUFFile, _ []byte, idx map[string]int) {
			g.Tensors[idx[prefix+"u"]].DType = GGMLTypeF32
		}},
		{"bad packed padding", "nonzero padding", func(g *GGUFFile, raw []byte, idx map[string]int) {
			raw[g.DataOffset+int(g.Tensors[idx[prefix+"u"]].Offset)+7] |= 0x80
		}},
		{"offset overflow", "byte range", func(g *GGUFFile, _ []byte, idx map[string]int) {
			g.Tensors[idx[prefix+"v2"]].Offset = math.MaxUint64
		}},
		{"rank overflow", "positive rank", func(g *GGUFFile, _ []byte, idx map[string]int) {
			g.Tensors[idx[prefix+"v1"]].Dims[0] = math.MaxUint64
		}},
		{"scale NaN", "nonfinite scale", func(g *GGUFFile, raw []byte, idx map[string]int) {
			binary.LittleEndian.PutUint16(raw[g.DataOffset+int(g.Tensors[idx[prefix+"v2"]].Offset):], 0x7e00)
		}},
		{"wrong parameter count", "original_parameters", func(g *GGUFFile, _ []byte, _ map[string]int) {
			g.Metadata[littleBitNamespace+"original_parameters"] = MetaValue{Kind: "u64", Value: uint64(1)}
		}},
		{"wrong bpw", "effective_bpw", func(g *GGUFFile, _ []byte, _ map[string]int) {
			g.Metadata[littleBitNamespace+"effective_bpw"] = MetaValue{Kind: "f64", Value: math.NaN()}
		}},
		{"fused QKV", "unsupported tensor", func(g *GGUFFile, _ []byte, idx map[string]int) {
			g.Tensors = append(g.Tensors, TensorInfo{Name: "blk.0.attn_qkv.weight"})
		}},
		{"factorized embeddings", "unsupported factorized projection", func(g *GGUFFile, _ []byte, idx map[string]int) {
			g.Tensors[idx[prefix+"u"]].Name = "token_embd.weight.lb.u"
		}},
		{"ordinary and factorized", "both ordinary and factorized", func(g *GGUFFile, _ []byte, _ map[string]int) {
			g.Tensors = append(g.Tensors, TensorInfo{Name: "blk.0.attn_q.weight"})
		}},
		{"unknown component", "unknown component", func(g *GGUFFile, _ []byte, idx map[string]int) {
			g.Tensors[idx[prefix+"u"]].Name = prefix + "typo"
		}},
	}
	data := writeLittleBitFixture(t, fixture.meta, fixture.packed)
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			raw := slices.Clone(data)
			g, err := ParseGGUFQuiet(raw)
			if err != nil {
				t.Fatal(err)
			}
			idx := make(map[string]int)
			for i, info := range g.Tensors {
				idx[info.Name] = i
			}
			tc.edit(g, raw, idx)
			if _, err = runnerFromParsedGGUF(raw, g, false, LoadOptions{}); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
	path := filepath.Join(t.TempDir(), "packed.gguf")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := CompressModel(path, filepath.Join(t.TempDir(), "q4.gguf"), CompressOptions{TargetFormat: GGMLTypeQ4_K}); err == nil ||
		!strings.Contains(err.Error(), "LittleBit") {
		t.Fatalf("compress error = %v", err)
	}
}
