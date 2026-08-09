package gopherllm

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func solidNRGBA(w, h int, r, g, b uint8) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	c := color.NRGBA{R: r, G: g, B: b, A: 255}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, c)
		}
	}
	return img
}

func TestRoundUpToMultiple(t *testing.T) {
	cases := []struct{ v, m, want int }{
		{0, 16, 0},
		{1, 16, 16},
		{16, 16, 16},
		{17, 16, 32},
		{100, 16, 112},
	}
	for _, c := range cases {
		if got := roundUpToMultiple(c.v, c.m); got != c.want {
			t.Errorf("roundUpToMultiple(%d,%d) = %d, want %d", c.v, c.m, got, c.want)
		}
	}
}

// TestPreprocessImagePixtralPatchGridShape checks the resize + round-up-to-
// patch-multiple math without downscaling (image already under maxEdge).
func TestPreprocessImagePixtralPatchGridShape(t *testing.T) {
	img := solidNRGBA(32, 32, 255, 0, 0)
	out, err := PreprocessImagePixtral(img, 16, 16, 1024, [3]float32{0.5, 0.4, 0.3}, [3]float32{0.25, 0.25, 0.25})
	if err != nil {
		t.Fatal(err)
	}
	if out.Rows != 2 || out.Cols != 2 {
		t.Fatalf("Rows/Cols = %d/%d, want 2/2", out.Rows, out.Cols)
	}
	if len(out.Pixels) != 4 {
		t.Fatalf("len(Pixels) = %d, want 4", len(out.Pixels))
	}
	for _, p := range out.Pixels {
		if len(p) != 3*16*16 {
			t.Fatalf("patch length = %d, want %d", len(p), 3*16*16)
		}
	}
}

// TestPreprocessImagePixtralDownscaleRoundsUpToPatchMultiple checks the
// aspect-preserving longest-edge downscale followed by round-up-to-multiple.
func TestPreprocessImagePixtralDownscaleRoundsUpToPatchMultiple(t *testing.T) {
	img := solidNRGBA(100, 50, 10, 20, 30)
	out, err := PreprocessImagePixtral(img, 8, 8, 32, [3]float32{0, 0, 0}, [3]float32{1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	// longest edge 100 -> scaled by 32/100 = 0.32: 100*0.32=32, 50*0.32=16,
	// both already multiples of 8.
	if out.Cols != 4 || out.Rows != 2 {
		t.Fatalf("Rows/Cols = %d/%d, want 2/4", out.Rows, out.Cols)
	}
}

// TestPreprocessImagePixtralRoundToExceedsPatchSize checks that roundTo
// (patchSize*spatialMergeSize in the real vision-tower pipeline) rounds the
// patch grid up to a multiple of the *merge* size, not just the patch size
// — a 14x14-patch image whose raw grid would otherwise be odd (7x7) must
// become an 8x8 grid so a 2x2 spatial merge divides it evenly.
func TestPreprocessImagePixtralRoundToExceedsPatchSize(t *testing.T) {
	const patchSize = 2
	const mergeSize = 2
	img := solidNRGBA(7*patchSize, 7*patchSize, 5, 6, 7) // a 7x7 patch grid at patchSize alone
	out, err := PreprocessImagePixtral(img, patchSize, patchSize*mergeSize, 1024, [3]float32{0, 0, 0}, [3]float32{1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	if out.Rows%mergeSize != 0 || out.Cols%mergeSize != 0 {
		t.Fatalf("Rows/Cols = %d/%d, want both divisible by merge size %d", out.Rows, out.Cols, mergeSize)
	}
	if out.Rows != 8 || out.Cols != 8 {
		t.Fatalf("Rows/Cols = %d/%d, want 8/8 (7 patches rounded up to the next even count)", out.Rows, out.Cols)
	}
}

func TestPixtralDynamicImageSizeMatchesMergedTokenGrid(t *testing.T) {
	// The reference Pixtral dynamic preprocessor rounds this common webcam
	// frame to 392x196 (14x7 merged tokens), not 392x224.
	if w, h := pixtralDynamicImageSize(384, 201, 28, 8*28*28, 1024*28*28); w != 392 || h != 196 {
		t.Fatalf("dynamic size = %dx%d, want 392x196", w, h)
	}
	img := solidNRGBA(384, 201, 12, 34, 56)
	pre, err := PreprocessImagePixtralDynamic(img, 14, 2, 8, 1024, [3]float32{}, [3]float32{1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	if pre.Rows != 14 || pre.Cols != 28 {
		t.Fatalf("dynamic patch grid = %dx%d, want 14x28", pre.Rows, pre.Cols)
	}
	if pre.Rows/2 != 7 || pre.Cols/2 != 14 {
		t.Fatalf("dynamic merged grid = %dx%d, want 7x14", pre.Rows/2, pre.Cols/2)
	}
}

func TestResizeBilinearNormalizedMatchesPixtralReference(t *testing.T) {
	// mtmd's Pixtral path uses align-corners interpolation and truncates the
	// intermediate RGB result to uint8. A pixel-center float resize would
	// produce different values at positions 1 and 2 here.
	img := image.NewNRGBA(image.Rect(0, 0, 3, 1))
	img.SetNRGBA(0, 0, color.NRGBA{R: 0, G: 10, B: 20, A: 255})
	img.SetNRGBA(1, 0, color.NRGBA{R: 100, G: 110, B: 120, A: 255})
	img.SetNRGBA(2, 0, color.NRGBA{R: 200, G: 210, B: 220, A: 255})
	got := resizeBilinearNormalized(img, 4, 1, [3]float32{}, [3]float32{1, 1, 1})
	want := [][3]uint8{{0, 10, 20}, {66, 76, 86}, {133, 143, 153}, {200, 210, 220}}
	if len(got) != len(want)*3 {
		t.Fatalf("resized values = %d, want %d", len(got), len(want)*3)
	}
	for x, pixel := range want {
		for c, raw := range pixel {
			if delta := math.Abs(float64(got[x*3+c] - float32(raw)/255)); delta > 1e-6 {
				t.Errorf("pixel %d channel %d = %.8f, want %.8f", x, c, got[x*3+c], float32(raw)/255)
			}
		}
	}
}

// TestPreprocessImagePixtralChannelLayout verifies the flattened patch
// vector is ordered [channel][kh][kw] with channel slowest-varying, matching
// the patch_embd weight's expected column order (pixtral_vision.go).
func TestPreprocessImagePixtralChannelLayout(t *testing.T) {
	const r, g, b = 200, 100, 50
	mean := [3]float32{0.5, 0.4, 0.3}
	std := [3]float32{0.25, 0.25, 0.25}
	img := solidNRGBA(2, 2, r, g, b)
	out, err := PreprocessImagePixtral(img, 2, 2, 1024, mean, std)
	if err != nil {
		t.Fatal(err)
	}
	if out.Rows != 1 || out.Cols != 1 || len(out.Pixels) != 1 {
		t.Fatalf("expected a single 1x1 patch grid, got Rows=%d Cols=%d len=%d", out.Rows, out.Cols, len(out.Pixels))
	}
	patch := out.Pixels[0]
	if len(patch) != 12 {
		t.Fatalf("patch length = %d, want 12", len(patch))
	}
	want := func(raw uint8, ch int) float32 {
		return (float32(raw)/255 - mean[ch]) / std[ch]
	}
	const eps = 1e-3
	checkBlock := func(name string, base int, raw uint8, ch int) {
		exp := want(raw, ch)
		for i := 0; i < 4; i++ {
			if v := patch[base+i]; float32(math.Abs(float64(v-exp))) > eps {
				t.Errorf("%s block[%d] = %v, want ~%v", name, i, v, exp)
			}
		}
	}
	checkBlock("R", 0, r, 0)
	checkBlock("G", 4, g, 1)
	checkBlock("B", 8, b, 2)
}

func TestPreprocessImagePixtralRejectsNonPositivePatchSize(t *testing.T) {
	img := solidNRGBA(4, 4, 1, 2, 3)
	if _, err := PreprocessImagePixtral(img, 0, 0, 1024, [3]float32{}, [3]float32{1, 1, 1}); err == nil {
		t.Fatal("expected an error for patchSize=0")
	}
}
