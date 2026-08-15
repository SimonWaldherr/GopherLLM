package gopherllm

import (
	"math"
	"testing"
)

// buildTinyPixtralVisionGGUFTyped is buildTinyPixtralVisionGGUF with a
// selectable weight dtype. Real mmproj files ship as F16, and F16 is the
// dtype whose loader branch differs most between the borrowed and owned
// paths, so the borrow tests below cover both rather than only the F32 form
// the demo fixture happens to use.
func buildTinyPixtralVisionGGUFTyped(dtype GGMLType) []byte {
	const (
		embLen    = 64
		heads     = 4
		headDim   = embLen / heads // 16
		ffnLen    = 64
		blocks    = 1
		patchSize = 4
		mergeSize = 2
		imageSize = 64
		projDim   = 256
	)
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "clip"},
		{"general.name", ggufStr, "tiny-vision"},
		{"clip.has_vision_encoder", ggufBool, true},
		{"clip.projector_type", ggufStr, "pixtral"},
		{"clip.use_silu", ggufBool, true},
		{"clip.vision.embedding_length", ggufU32, uint32(embLen)},
		{"clip.vision.feed_forward_length", ggufU32, uint32(ffnLen)},
		{"clip.vision.block_count", ggufU32, uint32(blocks)},
		{"clip.vision.attention.head_count", ggufU32, uint32(heads)},
		{"clip.vision.attention.head_dim", ggufU32, uint32(headDim)},
		{"clip.vision.attention.layer_norm_epsilon", ggufF32, float32(1e-5)},
		{"clip.vision.image_size", ggufU32, uint32(imageSize)},
		{"clip.vision.patch_size", ggufU32, uint32(patchSize)},
		{"clip.vision.spatial_merge_size", ggufU32, uint32(mergeSize)},
		{"clip.vision.projection_dim", ggufU32, uint32(projDim)},
	}

	// Weight values round-trip exactly through F16 (they are small multiples
	// of a power of two), so the borrowed-F16 and owned-F16 towers are
	// comparable bit for bit and a mismatch below means a wiring bug, not a
	// precision difference.
	body := func(n, seed int) []byte {
		vals := make([]float32, n)
		for i := range vals {
			vals[i] = float32((i+seed)%17-8) / 64
		}
		if dtype == GGMLTypeF32 {
			return f32Bytes(vals)
		}
		raw := make([]byte, n*2)
		for i, v := range vals {
			h := F32ToF16(v)
			raw[i*2], raw[i*2+1] = byte(h), byte(h>>8)
		}
		return raw
	}
	mat := func(name string, rows, cols, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: dtype, data: body(rows*cols, seed)}
	}
	matDims := func(name string, dims []uint64, seed int) ggufTensor {
		n := 1
		for _, d := range dims {
			n *= int(d)
		}
		return ggufTensor{name: name, dims: dims, dtype: dtype, data: body(n, seed)}
	}
	// Norm vectors stay F32: loadF32Vec reads them directly and the real
	// checkpoint stores them that way too (see the mmproj tensor listing).
	vec := func(name string, n, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(onesF32(n))}
	}

	mergedLen := embLen * mergeSize * mergeSize

	tensors := []ggufTensor{
		// Four dimensions on purpose: patch_embd is a stride==kernel conv2d,
		// stored [patch, patch, 3, embLen]. loadWeight derives Rows/Cols from
		// Dims[1]/Dims[0], which for this tensor are the two spatial axes, not
		// the [embLen, 3*patch*patch] matrix the encoder multiplies by.
		matDims("v.patch_embd.weight", []uint64{uint64(patchSize), uint64(patchSize), 3, uint64(embLen)}, 1),
		vec("v.pre_ln.weight", embLen, 2),
		mat("v.blk.0.attn_q.weight", embLen, embLen, 3),
		mat("v.blk.0.attn_k.weight", embLen, embLen, 4),
		mat("v.blk.0.attn_v.weight", embLen, embLen, 5),
		mat("v.blk.0.attn_out.weight", embLen, embLen, 6),
		vec("v.blk.0.ln1.weight", embLen, 7),
		mat("v.blk.0.ffn_gate.weight", ffnLen, embLen, 8),
		mat("v.blk.0.ffn_up.weight", ffnLen, embLen, 9),
		mat("v.blk.0.ffn_down.weight", embLen, ffnLen, 10),
		vec("v.blk.0.ln2.weight", embLen, 11),
		vec("mm.input_norm.weight", embLen, 12),
		mat("mm.patch_merger.weight", embLen, mergedLen, 13),
		mat("mm.1.weight", projDim, embLen, 14),
		mat("mm.2.weight", projDim, projDim, 15),
		vec("v.token_embd.img_break", projDim, 16),
	}
	return buildGGUF(3, kvs, tensors)
}

// TestPixtralVisionBorrowedMatchesOwned pins the two loader modes against each
// other.
//
// LoadOptions.BorrowQuantized selects loadPixtralVisionModel's borrowing path,
// which keeps scalar (F32/F16) tensors in their on-disk form instead of
// expanding them into owned []float32. That mode is not an obscure corner: the
// WASM bridge is the only caller in the tree that sets BorrowQuantized (see
// cmd/gopherllm-wasm/bridge.go), so it is exactly and only what runs when a
// user loads a model "directly in the browser" — while every native path, and
// every other vision test here, exercises the owned path. A wiring bug
// confined to the borrowed tower is therefore invisible everywhere except in
// a browser tab, which is the worst place to discover it.
//
// The two modes must agree numerically, so this compares embeddings rather
// than just shapes: a wrong Rows/Cols on a borrowed weight silently truncates
// or reshapes a projection instead of failing loudly.
func TestPixtralVisionBorrowedMatchesOwned(t *testing.T) {
	for _, dtype := range []GGMLType{GGMLTypeF32, GGMLTypeF16} {
		t.Run(dtype.String(), func(t *testing.T) {
			data := buildTinyPixtralVisionGGUFTyped(dtype)
			gguf, err := ParseGGUFQuiet(data)
			if err != nil {
				t.Fatal(err)
			}

			ownedCfg, ownedW, err := loadPixtralVisionModel(data, gguf, false, false, nil)
			if err != nil {
				t.Fatalf("owned load: %v", err)
			}
			borrowCfg, borrowW, err := loadPixtralVisionModel(data, gguf, false, true, nil)
			if err != nil {
				t.Fatalf("borrowed load: %v", err)
			}
			if ownedCfg != borrowCfg {
				t.Fatalf("config differs between modes:\n owned    %+v\n borrowed %+v", ownedCfg, borrowCfg)
			}

			// A 4x4 patch grid: big enough for two merge windows per axis, so
			// the spatial merge and the patch-embedding projection both get
			// exercised over more than a single trivial window.
			img := &PreprocessedImage{Rows: 4, Cols: 4}
			patchLen := 3 * ownedCfg.PatchSize * ownedCfg.PatchSize
			for i := range 16 {
				p := make([]float32, patchLen)
				for j := range p {
					p[j] = float32((i*7+j)%13-6) / 8
				}
				img.Pixels = append(img.Pixels, p)
			}

			ownedEmb, ownedRows, ownedCols, err := EncodeImagePixtral(ownedCfg, ownedW, img)
			if err != nil {
				t.Fatalf("owned encode: %v", err)
			}
			borrowEmb, borrowRows, borrowCols, err := EncodeImagePixtral(borrowCfg, borrowW, img)
			if err != nil {
				t.Fatalf("borrowed encode: %v", err)
			}
			if ownedRows != borrowRows || ownedCols != borrowCols {
				t.Fatalf("merged grid differs: owned %dx%d, borrowed %dx%d", ownedRows, ownedCols, borrowRows, borrowCols)
			}
			if len(ownedEmb) != len(borrowEmb) {
				t.Fatalf("embedding count differs: owned %d, borrowed %d", len(ownedEmb), len(borrowEmb))
			}
			for i := range ownedEmb {
				if len(borrowEmb[i]) != ownedCfg.ProjectionDim {
					t.Fatalf("borrowed embed %d has width %d, want ProjectionDim %d", i, len(borrowEmb[i]), ownedCfg.ProjectionDim)
				}
				for j := range ownedEmb[i] {
					got, want := borrowEmb[i][j], ownedEmb[i][j]
					if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
						t.Fatalf("borrowed embed %d component %d is %v", i, j, got)
					}
					if diff := math.Abs(float64(got - want)); diff > 1e-4*math.Max(1, math.Abs(float64(want))) {
						t.Fatalf("borrowed embed %d component %d = %v, owned = %v (diff %g)", i, j, got, want, diff)
					}
				}
			}
		})
	}
}

// TestPixtralVisionBorrowedWeightShapes checks the loaded tensor geometry
// directly, so a failure names the tensor that is wired wrong instead of only
// showing up as a numerical mismatch several matmuls later.
//
// v.patch_embd.weight is the one that matters: GGUF stores it with four
// dimensions ([patch, patch, 3, embLen]) because it is a conv2d kernel, but
// the encoder multiplies by it as an ordinary [embLen, 3*patch*patch] linear
// layer. loadWeight's generic Rows=Dims[1]/Cols=Dims[0] rule describes the
// conv layout, not the matrix, so a borrowed (raw) patch_embd carries the two
// spatial axes as its shape unless the loader corrects them.
func TestPixtralVisionBorrowedWeightShapes(t *testing.T) {
	data := buildTinyPixtralVisionGGUFTyped(GGMLTypeF16)
	gguf, err := ParseGGUFQuiet(data)
	if err != nil {
		t.Fatal(err)
	}
	cfg, w, err := loadPixtralVisionModel(data, gguf, false, true, nil)
	if err != nil {
		t.Fatal(err)
	}

	dim := cfg.EmbeddingLength
	merged := dim * cfg.SpatialMergeSize * cfg.SpatialMergeSize
	for _, tc := range []struct {
		name       string
		w          Weight
		rows, cols int
	}{
		{"v.patch_embd.weight", w.PatchEmbd, dim, 3 * cfg.PatchSize * cfg.PatchSize},
		{"v.blk.0.attn_q.weight", w.Layers[0].Q, dim, dim},
		{"v.blk.0.ffn_gate.weight", w.Layers[0].FFNGate, cfg.FeedForwardLength, dim},
		{"v.blk.0.ffn_down.weight", w.Layers[0].FFNDown, dim, cfg.FeedForwardLength},
		{"mm.patch_merger.weight", w.PatchMerger, dim, merged},
		{"mm.1.weight", w.Proj1, cfg.ProjectionDim, dim},
		{"mm.2.weight", w.Proj2, cfg.ProjectionDim, cfg.ProjectionDim},
	} {
		if tc.w.F32 != nil {
			// An owned copy carries its shape implicitly; only raw weights
			// need Rows/Cols to be right.
			continue
		}
		if tc.w.Rows != tc.rows || tc.w.Cols != tc.cols {
			t.Errorf("%s borrowed as Rows=%d Cols=%d, want Rows=%d Cols=%d",
				tc.name, tc.w.Rows, tc.w.Cols, tc.rows, tc.cols)
		}
		if want := tc.rows * tc.cols * scalarBytesPerElement(tc.w.Type); len(tc.w.Raw) < want {
			t.Errorf("%s borrowed raw is %d bytes, want at least %d", tc.name, len(tc.w.Raw), want)
		}
	}
}
