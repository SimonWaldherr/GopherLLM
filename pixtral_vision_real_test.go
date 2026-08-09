package gopherllm

import (
	"image"
	"image/color"
	"math"
	"os"
	"testing"
)

// TestPixtralRope2DAlternatesAxisFrequencies locks down Pixtral's distinct
// row/column 2D-RoPE frequency layout. The reference creates theta^(-2i/D),
// then assigns its even entries to rows and odd entries to columns before
// applying adjacent-pair RoPE independently within each axis half.
func TestPixtralRope2DAlternatesAxisFrequencies(t *testing.T) {
	const (
		headDim = 8
		theta   = float32(10000)
		row     = 3
		col     = 5
	)
	rowInv, colInv := buildPixtralRope2DInvFreq(headDim, theta)
	if len(rowInv) != headDim/4 || len(colInv) != headDim/4 {
		t.Fatalf("frequency lengths = %d/%d, want %d/%d", len(rowInv), len(colInv), headDim/4, headDim/4)
	}
	for j := range rowInv {
		wantRow := float32(1 / math.Pow(float64(theta), float64(4*j)/headDim))
		wantCol := float32(1 / math.Pow(float64(theta), float64(4*j+2)/headDim))
		if math.Abs(float64(rowInv[j]-wantRow)) > 1e-6 {
			t.Errorf("rowInv[%d] = %v, want %v", j, rowInv[j], wantRow)
		}
		if math.Abs(float64(colInv[j]-wantCol)) > 1e-6 {
			t.Errorf("colInv[%d] = %v, want %v", j, colInv[j], wantCol)
		}
		if rowInv[j] == colInv[j] {
			t.Errorf("rowInv[%d] and colInv[%d] must use alternating source frequencies, both are %v", j, j, rowInv[j])
		}
	}

	var sin, cos []float32
	half, pairs := preparePixtralRope2DScratch(row, col, headDim, rowInv, colInv, &sin, &cos)
	if half != headDim/2 || pairs != headDim/2 {
		t.Fatalf("half/pairs = %d/%d, want %d/%d", half, pairs, headDim/2, headDim/2)
	}
	for j := range rowInv {
		wantRowSin, wantRowCos := math.Sincos(float64(float32(row) * rowInv[j]))
		wantColSin, wantColCos := math.Sincos(float64(float32(col) * colInv[j]))
		if math.Abs(float64(sin[j])-wantRowSin) > 1e-6 || math.Abs(float64(cos[j])-wantRowCos) > 1e-6 {
			t.Errorf("row angle %d = sin/cos %v/%v, want %v/%v", j, sin[j], cos[j], wantRowSin, wantRowCos)
		}
		if math.Abs(float64(sin[len(rowInv)+j])-wantColSin) > 1e-6 || math.Abs(float64(cos[len(rowInv)+j])-wantColCos) > 1e-6 {
			t.Errorf("column angle %d = sin/cos %v/%v, want %v/%v", j, sin[len(rowInv)+j], cos[len(rowInv)+j], wantColSin, wantColCos)
		}
	}

	// The combined table is consumed as adjacent pairs in each half of a
	// vision-attention head: (0,1), (2,3) for rows, then (4,5), (6,7) for
	// columns. A split-half rotation would incorrectly pair row values with
	// column values instead.
	vec := []float32{1, 0, 1, 0, 1, 0, 1, 0}
	applyPreparedRope(vec, headDim, 1, half, pairs, sin, cos, true)
	for i := 0; i < headDim/2; i++ {
		if i%2 == 0 {
			if math.Abs(float64(vec[i]-cos[i/2])) > 1e-6 {
				t.Errorf("interleaved pair %d first component = %v, want cos=%v", i/2, vec[i], cos[i/2])
			}
		} else if math.Abs(float64(vec[i]-sin[i/2])) > 1e-6 {
			t.Errorf("interleaved pair %d second component = %v, want sin=%v", i/2, vec[i], sin[i/2])
		}
	}
	for i := headDim / 2; i < headDim; i++ {
		pair := i / 2
		if i%2 == 0 {
			if math.Abs(float64(vec[i]-cos[pair])) > 1e-6 {
				t.Errorf("interleaved column pair %d first component = %v, want cos=%v", pair, vec[i], cos[pair])
			}
		} else if math.Abs(float64(vec[i]-sin[pair])) > 1e-6 {
			t.Errorf("interleaved column pair %d second component = %v, want sin=%v", pair, vec[i], sin[pair])
		}
	}
}

// TestPixtralVisionRealMMProjLoadsAndEncodes is an opt-in plausibility check
// against a real downloaded mmproj GGUF (no text model needed -- this only
// exercises LoadPixtralVisionModel/EncodeImagePixtral in isolation). It
// checks shapes and numerical sanity (no NaN/Inf, non-degenerate output),
// not semantic correctness -- see the project plan's M1.12 for why a
// stronger check isn't automatable without depending on llama.cpp.
//
//	GOPHERLLM_VISION_MMPROJ=mmproj-F16.gguf go test -run TestPixtralVisionRealMMProjLoadsAndEncodes -v .
func TestPixtralVisionRealMMProjLoadsAndEncodes(t *testing.T) {
	path := os.Getenv("GOPHERLLM_VISION_MMPROJ")
	if path == "" {
		t.Skip("set GOPHERLLM_VISION_MMPROJ=<path to mmproj gguf> to run this test")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	gguf, err := ParseGGUF(data)
	if err != nil {
		t.Fatal(err)
	}
	vc, weights, err := LoadPixtralVisionModel(data, gguf, false, os.Stderr)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("config: %+v", vc)
	if vc.EmbeddingLength <= 0 || vc.BlockCount <= 0 || len(weights.Layers) != vc.BlockCount {
		t.Fatalf("implausible config/weights: %+v, %d layers loaded", vc, len(weights.Layers))
	}
	if len(weights.ImgBreak) != vc.ProjectionDim {
		t.Fatalf("ImgBreak width = %d, want %d (ProjectionDim)", len(weights.ImgBreak), vc.ProjectionDim)
	}

	// A small solid-color image, well under ImageSize, that still exercises
	// multiple merge windows (so the spatial-merge indexing gets covered by
	// more than a trivial 1x1 case).
	img := image.NewNRGBA(image.Rect(0, 0, 4*vc.PatchSize*vc.SpatialMergeSize, 3*vc.PatchSize*vc.SpatialMergeSize))
	c := color.NRGBA{R: 30, G: 160, B: 220, A: 255} // an arbitrary, unambiguous blue
	for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
		for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
			img.SetNRGBA(x, y, c)
		}
	}

	pre, err := PreprocessImagePixtral(img, vc.PatchSize, vc.PatchSize*vc.SpatialMergeSize, vc.ImageSize, vc.ImageMean, vc.ImageStd)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("preprocessed patch grid: %dx%d", pre.Rows, pre.Cols)

	embeds, mergedRows, mergedCols, err := EncodeImagePixtral(vc, weights, pre)
	if err != nil {
		t.Fatal(err)
	}
	wantRows, wantCols := pre.Rows/vc.SpatialMergeSize, pre.Cols/vc.SpatialMergeSize
	if mergedRows != wantRows || mergedCols != wantCols {
		t.Fatalf("merged grid = %dx%d, want %dx%d", mergedRows, mergedCols, wantRows, wantCols)
	}
	if len(embeds) != mergedRows*mergedCols {
		t.Fatalf("len(embeds) = %d, want %d", len(embeds), mergedRows*mergedCols)
	}
	for i, e := range embeds {
		if len(e) != vc.ProjectionDim {
			t.Fatalf("embeds[%d] width = %d, want %d", i, len(e), vc.ProjectionDim)
		}
		var sumSq float64
		for _, v := range e {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("embeds[%d] contains NaN/Inf", i)
			}
			sumSq += float64(v) * float64(v)
		}
		norm := math.Sqrt(sumSq)
		if norm < 1e-6 {
			t.Fatalf("embeds[%d] is degenerate (near-zero norm %v)", i, norm)
		}
		if i == 0 {
			t.Logf("embeds[0] L2 norm = %v, first 4 values = %v", norm, e[:4])
		}
	}

	// Since every patch is the identical solid color, every merged output
	// embedding should be (near-)identical too -- a real, cheap correctness
	// signal beyond "no NaN": if the merge/attention wiring were scrambling
	// patch identity, uniform input wouldn't produce uniform output.
	for i := 1; i < len(embeds); i++ {
		var maxDiff float32
		for j := range embeds[0] {
			d := embeds[0][j] - embeds[i][j]
			if d < 0 {
				d = -d
			}
			if d > maxDiff {
				maxDiff = d
			}
		}
		if maxDiff > 1e-2 {
			t.Fatalf("embeds[%d] differs from embeds[0] by up to %v on a uniform solid-color image (want near-identical)", i, maxDiff)
		}
	}
}
