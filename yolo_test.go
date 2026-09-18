package gopherllm

import (
	"image"
	"image/color"
	"testing"
)

func TestPrepareYOLOImageLetterboxesCHW(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 4, 2))
	img.SetRGBA(0, 0, color.RGBA{R: 255, G: 128, B: 64, A: 255})
	pre, err := PrepareYOLOImage(img, YOLOConfig{InputWidth: 8, InputHeight: 8})
	if err != nil {
		t.Fatal(err)
	}
	if pre.Scale != 2 || pre.PadX != 0 || pre.PadY != 2 {
		t.Fatalf("mapping = scale %v, pad (%v,%v)", pre.Scale, pre.PadX, pre.PadY)
	}
	if got := pre.Pixels[0]; got != 114.0/255 {
		t.Fatalf("top padding = %v", got)
	}
	if got := pre.Pixels[2*8]; got == 114.0/255 {
		t.Fatal("resized image was not written after padding")
	}
	if got := pre.Pixels[64+2*8]; got < .49 || got > .51 {
		t.Fatalf("green CHW pixel = %v", got)
	}
}

func TestDecodeYOLOv8MapsAndSuppressesBoxes(t *testing.T) {
	input := YOLOImage{Width: 640, Height: 640, SourceWidth: 320, SourceHeight: 160, Scale: 2, PadY: 160}
	// [features, anchors]: boxes 100,200,80,40 and 102,200,80,40 both map
	// to source boxes; the second overlaps the first, while the third is class 1.
	out := []float32{
		100, 102, 300, // cx
		200, 200, 200, // cy
		80, 80, 40, // width
		40, 40, 40, // height
		.9, .8, .1, // class 0
		.1, .1, .95, // class 1
	}
	dets, err := DecodeYOLOv8(out, 6, 3, input, YOLOConfig{ScoreThreshold: .25, IoUThreshold: .45, ClassNames: []string{"person", "car"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(dets) != 2 {
		t.Fatalf("detections = %#v", dets)
	}
	if dets[0].ClassID != 1 || dets[0].Label != "car" {
		t.Fatalf("best detection = %#v", dets[0])
	}
	if dets[1].ClassID != 0 || dets[1].Box.YMin != 10 || dets[1].Box.YMax != 30 {
		t.Fatalf("mapped detection = %#v", dets[1])
	}
}

func TestDecodeYOLOv8AcceptsTransposedOutput(t *testing.T) {
	input := YOLOImage{Width: 64, Height: 64, SourceWidth: 64, SourceHeight: 64, Scale: 1}
	// [anchors, features]
	dets, err := DecodeYOLOv8([]float32{32, 32, 10, 10, .8, .1}, 1, 6, input, YOLOConfig{ScoreThreshold: .5})
	if err != nil {
		t.Fatal(err)
	}
	if len(dets) != 1 || dets[0].ClassID != 0 {
		t.Fatalf("detections = %#v", dets)
	}
}
