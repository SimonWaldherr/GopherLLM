package gopherllm

import (
	"image"
	"image/color"
	"math"
	"os"
	"testing"
)

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
