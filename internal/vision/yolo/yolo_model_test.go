package yolo

import (
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/SimonWaldherr/GopherLLM/internal/yolotest"
)

// Tensor renamings for yolotest.Checkpoint.Safetensors.
var (
	yoloCandleNames      = yoloWeightSource{candle: true}.resolve
	yoloUltralyticsNames = yoloWeightSource{prefix: "model."}.resolve
)

func yoloSynthImage() image.Image {
	img := image.NewRGBA(image.Rect(0, 0, 90, 60))
	for y := range 60 {
		for x := range 90 {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x * 2), G: uint8(y * 4), B: uint8((x + y) * 3), A: 255})
		}
	}
	return img
}

func TestYOLOv8LoadsBothNamingSchemesIdentically(t *testing.T) {
	candle, err := LoadYOLOSafetensors(yolotest.V8().Safetensors(yoloCandleNames))
	if err != nil {
		t.Fatal(err)
	}
	ultra, err := LoadYOLOSafetensors(yolotest.V8().Safetensors(yoloUltralyticsNames))
	if err != nil {
		t.Fatal(err)
	}
	if candle.NumClasses != 3 || candle.regMax != 16 {
		t.Fatalf("inferred %d classes, regMax %d; want 3, 16", candle.NumClasses, candle.regMax)
	}
	input, err := PrepareYOLOImage(yoloSynthImage(), YOLOConfig{InputWidth: 64, InputHeight: 96})
	if err != nil {
		t.Fatal(err)
	}
	a, rows, cols, err := candle.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	// 64x96 input: (8x12)+(4x6)+(2x3) anchors.
	if rows != 7 || cols != 96+24+6 {
		t.Fatalf("output shape [%d,%d], want [7,126]", rows, cols)
	}
	b, _, _, err := ultra.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("output %d differs between namings: %v vs %v", i, a[i], b[i])
		}
		if math.IsNaN(float64(a[i])) || math.IsInf(float64(a[i]), 0) {
			t.Fatalf("output %d is %v", i, a[i])
		}
	}
	for c := 4; c < rows; c++ {
		for j := range cols {
			if s := a[c*cols+j]; s < 0 || s > 1 {
				t.Fatalf("class score %v outside [0,1]", s)
			}
		}
	}
	// Box widths/heights are sums of DFL expectations in [0, regMax-1]
	// grid cells, scaled by the level stride.
	for j := range cols {
		stride := float32(8)
		if j >= 96+24 {
			stride = 32
		} else if j >= 96 {
			stride = 16
		}
		if w := a[2*cols+j]; w < 0 || w > 2*15*stride {
			t.Fatalf("anchor %d width %v outside DFL range", j, w)
		}
	}
}

func TestYOLOv8DetectReturnsSourceCoordinates(t *testing.T) {
	m, err := LoadYOLOSafetensors(yolotest.V8().Safetensors(yoloCandleNames))
	if err != nil {
		t.Fatal(err)
	}
	m.ClassNames = []string{"a", "b", "c"}
	dets, err := m.Detect(yoloSynthImage(), YOLOConfig{InputWidth: 64, InputHeight: 64, ScoreThreshold: 0.01})
	if err != nil {
		t.Fatal(err)
	}
	if len(dets) == 0 {
		t.Fatal("no detections at a 0.01 threshold from random weights")
	}
	for _, d := range dets {
		if d.Label == "" || d.Box.XMin < 0 || d.Box.YMin < 0 || d.Box.XMax > 90 || d.Box.YMax > 60 {
			t.Fatalf("detection %+v not labelled or outside the 90x60 source", d)
		}
	}
	if _, err := m.Detect(yoloSynthImage(), YOLOConfig{InputWidth: 60, InputHeight: 64}); err == nil {
		t.Fatal("expected an error for an input width that is not a multiple of 32")
	}
}

func TestYOLOConvMatchesDirectConvolution(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, tc := range []struct{ in, out, k, stride, h, w int }{
		{3, 5, 3, 2, 9, 7}, {4, 6, 3, 1, 5, 8}, {6, 7, 1, 1, 4, 3},
	} {
		cv := yoloConv{in: tc.in, out: tc.out, k: tc.k, stride: tc.stride, act: true,
			w: make([]float32, tc.out*tc.in*tc.k*tc.k), b: make([]float32, tc.out)}
		for i := range cv.w {
			cv.w[i] = float32(rng.NormFloat64())
		}
		for i := range cv.b {
			cv.b[i] = float32(rng.NormFloat64())
		}
		x := yoloTensor{c: tc.in, h: tc.h, w: tc.w, data: make([]float32, tc.in*tc.h*tc.w)}
		for i := range x.data {
			x.data[i] = float32(rng.NormFloat64())
		}
		got, err := cv.forward(x)
		if err != nil {
			t.Fatal(err)
		}
		want, oh, ow := referenceYOLOConv(x, cv)
		if got.h != oh || got.w != ow {
			t.Fatalf("%+v: output %dx%d, want %dx%d", tc, got.w, got.h, ow, oh)
		}
		for i, v := range want {
			v = v / (1 + float32(math.Exp(float64(-v))))
			if d := math.Abs(float64(got.data[i] - v)); d > 1e-4 {
				t.Fatalf("%+v: output %d = %v, want %v", tc, i, got.data[i], v)
			}
		}
	}
}

func referenceYOLOConv(x yoloTensor, cv yoloConv) ([]float32, int, int) {
	pad := cv.k / 2
	oh := (x.h+2*pad-cv.k)/cv.stride + 1
	ow := (x.w+2*pad-cv.k)/cv.stride + 1
	out := make([]float32, cv.out*oh*ow)
	for o := range cv.out {
		for y := range oh {
			for xout := range ow {
				var sum float32
				for c := range cv.in {
					for ky := range cv.k {
						for kx := range cv.k {
							sy, sx := y*cv.stride+ky-pad, xout*cv.stride+kx-pad
							if sy < 0 || sy >= x.h || sx < 0 || sx >= x.w {
								continue
							}
							wi := ((o*cv.in+c)*cv.k+ky)*cv.k + kx
							sum += cv.w[wi] * x.data[c*x.h*x.w+sy*x.w+sx]
						}
					}
				}
				out[o*oh*ow+y*ow+xout] = sum + cv.b[o]
			}
		}
	}
	return out, oh, ow
}

func TestYOLOPortableGEMMMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	m, n, k := 7, 1100, 13
	a, b := make([]float32, m*k), make([]float32, k*n)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	for i := range b {
		b[i] = float32(rng.NormFloat64())
	}
	got := make([]float32, m*n)
	for i := range got {
		got[i] = 99 // must be overwritten, not accumulated into
	}
	yoloGEMMPortable(m, n, k, a, b, got)
	fast := make([]float32, m*n)
	yoloGEMM(m, n, k, a, b, fast)
	for i := range m {
		for j := range n {
			var want float64
			for p := range k {
				want += float64(a[i*k+p]) * float64(b[p*n+j])
			}
			if math.Abs(float64(got[i*n+j])-want) > 1e-4 || math.Abs(float64(fast[i*n+j])-want) > 1e-4 {
				t.Fatalf("c[%d,%d] = %v (portable) / %v (dispatch), want %v", i, j, got[i*n+j], fast[i*n+j], want)
			}
		}
	}
}

func TestYOLOMaxPoolSame(t *testing.T) {
	x := yoloTensor{c: 1, h: 3, w: 4, data: []float32{
		1, 2, 3, 4,
		5, 0, 0, 0,
		0, 0, 9, 0,
	}}
	got := yoloMaxPool(x, 3)
	want := []float32{5, 5, 4, 4, 5, 9, 9, 9, 5, 9, 9, 9}
	for i := range want {
		if got.data[i] != want[i] {
			t.Fatalf("pooled = %v, want %v", got.data, want)
		}
	}
}

// TestYOLOv8RealCheckpoint runs a real checkpoint when one is supplied, e.g.
//
//	GOPHERLLM_YOLO_MODEL=yolov8n.safetensors GOPHERLLM_YOLO_IMAGE=bike.jpg go test -run RealCheckpoint -v
func TestYOLOv8RealCheckpoint(t *testing.T) {
	modelPath, imagePath := os.Getenv("GOPHERLLM_YOLO_MODEL"), os.Getenv("GOPHERLLM_YOLO_IMAGE")
	if modelPath == "" || imagePath == "" {
		t.Skip("set GOPHERLLM_YOLO_MODEL and GOPHERLLM_YOLO_IMAGE to run against a real checkpoint")
	}
	m, err := LoadYOLO(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if m.NumClasses == 80 {
		m.ClassNames = COCOClassNames
	}
	f, err := os.Open(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	dets, err := m.Detect(img, DefaultYOLOConfig())
	if err != nil {
		t.Fatal(err)
	}
	if len(dets) == 0 {
		t.Fatal("no detections")
	}
	for _, d := range dets {
		t.Logf("%-14s %.3f  [%.0f %.0f %.0f %.0f]", d.Label, d.Confidence, d.Box.XMin, d.Box.YMin, d.Box.XMax, d.Box.YMax)
	}
}

func TestYOLO11LoadsAndRuns(t *testing.T) {
	m, err := LoadYOLOSafetensors(yolotest.V11().Safetensors(yoloUltralyticsNames))
	if err != nil {
		t.Fatal(err)
	}
	if m.Version != "yolo11" || m.NumClasses != 3 {
		t.Fatalf("loaded %s with %d classes, want yolo11 with 3", m.Version, m.NumClasses)
	}
	psa := m.layers[10].block.(*yoloC2PSA)
	if a := psa.m[0].attn; a.heads != 2 || a.headDim != 64 || a.keyDim != 32 {
		t.Fatalf("attention geometry heads=%d headDim=%d keyDim=%d, want 2/64/32", a.heads, a.headDim, a.keyDim)
	}
	if _, ok := m.layers[6].block.(*yoloC2f).m[0].(*yoloC3k); !ok {
		t.Fatal("layer 6 should hold a C3k inner module")
	}
	if !m.classes[0][0].depthwise {
		t.Fatal("YOLO11 class branch should start with a depthwise convolution")
	}
	input, err := PrepareYOLOImage(yoloSynthImage(), YOLOConfig{InputWidth: 96, InputHeight: 64})
	if err != nil {
		t.Fatal(err)
	}
	out, rows, cols, err := m.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	if rows != 7 || cols != 12*8+6*4+3*2 {
		t.Fatalf("output shape [%d,%d], want [7,126]", rows, cols)
	}
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output %d is %v", i, v)
		}
	}
}

func TestYOLOUnknownLayoutIsRejected(t *testing.T) {
	s := yolotest.New()
	s.Conv("0", 4, 3, 3)
	if _, err := LoadYOLOSafetensors(s.Safetensors(yoloUltralyticsNames)); err == nil {
		t.Fatal("expected an error for a checkpoint that is neither YOLOv8 nor YOLO11")
	}
}

// TestYOLOAttentionMatchesReference checks the PSA attention against a
// float64 transcription of Ultralytics' Attention.forward, including the
// per-head q/k/v channel packing and the depthwise positional term.
func TestYOLOAttentionMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const dim, heads, keyDim, h, w = 8, 2, 2, 3, 4
	headDim, n := dim/heads, h*w
	randConv := func(out, in, k int, depthwise bool) yoloConv {
		c := yoloConv{out: out, in: in, k: k, stride: 1, depthwise: depthwise, b: make([]float32, out)}
		per := in * k * k
		if depthwise {
			per = k * k
		}
		c.w = make([]float32, out*per)
		for i := range c.w {
			c.w[i] = float32(rng.NormFloat64())
		}
		for i := range c.b {
			c.b[i] = float32(rng.NormFloat64())
		}
		return c
	}
	a := yoloAttention{
		qkv: randConv(dim+2*heads*keyDim, dim, 1, false), proj: randConv(dim, dim, 1, false),
		pe: randConv(dim, dim, 3, true), heads: heads, keyDim: keyDim, headDim: headDim,
	}
	x := yoloTensor{c: dim, h: h, w: w, data: make([]float32, dim*n)}
	for i := range x.data {
		x.data[i] = float32(rng.NormFloat64())
	}
	got, err := a.forward(x)
	if err != nil {
		t.Fatal(err)
	}

	conv1x1 := func(c yoloConv, in []float64) []float64 {
		out := make([]float64, c.out*n)
		for o := range c.out {
			for p := range n {
				s := float64(c.b[o])
				for i := range c.in {
					s += float64(c.w[o*c.in+i]) * in[i*n+p]
				}
				out[o*n+p] = s
			}
		}
		return out
	}
	xs := make([]float64, len(x.data))
	for i, v := range x.data {
		xs[i] = float64(v)
	}
	qkv := conv1x1(a.qkv, xs)
	// qkv.view(heads, 2*keyDim+headDim, N).split([keyDim, keyDim, headDim])
	per := 2*keyDim + headDim
	at := func(head, ch, p int) float64 { return qkv[(head*per+ch)*n+p] }
	v := make([]float64, dim*n)
	attended := make([]float64, dim*n)
	for hd := range heads {
		for d := range headDim {
			for p := range n {
				v[(hd*headDim+d)*n+p] = at(hd, 2*keyDim+d, p)
			}
		}
		for i := range n {
			scores := make([]float64, n)
			var largest = math.Inf(-1)
			for j := range n {
				for d := range keyDim {
					scores[j] += at(hd, d, i) * at(hd, keyDim+d, j)
				}
				scores[j] /= math.Sqrt(keyDim)
				largest = math.Max(largest, scores[j])
			}
			var sum float64
			for j := range scores {
				scores[j] = math.Exp(scores[j] - largest)
				sum += scores[j]
			}
			for d := range headDim {
				var acc float64
				for j := range n {
					acc += at(hd, 2*keyDim+d, j) * scores[j] / sum
				}
				attended[(hd*headDim+d)*n+i] = acc
			}
		}
	}
	for c := range dim {
		for y := range h {
			for xx := range w {
				s := float64(a.pe.b[c])
				for ky := range 3 {
					for kx := range 3 {
						iy, ix := y+ky-1, xx+kx-1
						if iy >= 0 && iy < h && ix >= 0 && ix < w {
							s += float64(a.pe.w[c*9+ky*3+kx]) * v[c*n+iy*w+ix]
						}
					}
				}
				attended[c*n+y*w+xx] += s
			}
		}
	}
	want := conv1x1(a.proj, attended)
	for i := range want {
		if math.Abs(float64(got.data[i])-want[i]) > 1e-3 {
			t.Fatalf("attention output %d = %v, want %v", i, got.data[i], want[i])
		}
	}
}
