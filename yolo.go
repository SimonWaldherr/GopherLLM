package gopherllm

import (
	"fmt"
	"image"
	"math"
	"sort"
)

// YOLOConfig controls native YOLO image preparation and post-processing.
// InputWidth and InputHeight are normally 640. ClassNames is optional; when
// present it supplies human-readable labels for Detection.Label.
//
// This package deliberately does not depend on an ONNX runtime. A native
// convolution executor can pass its final output tensor to DecodeYOLOv8.
type YOLOConfig struct {
	InputWidth, InputHeight int
	ClassNames              []string
	ScoreThreshold          float32
	IoUThreshold            float32
	MaxDetections           int
}

// DefaultYOLOConfig returns conservative defaults compatible with common
// YOLOv8 detection exports. Model-specific class labels remain the caller's
// responsibility: COCO labels are not assumed for a custom-trained model.
func DefaultYOLOConfig() YOLOConfig {
	return YOLOConfig{InputWidth: 640, InputHeight: 640, ScoreThreshold: 0.25, IoUThreshold: 0.45, MaxDetections: 300}
}

// YOLOImage is a letterboxed, CHW RGB image. Pixels contains three contiguous
// planes in the [0,1] range, so it can be supplied directly to a native YOLO
// convolution graph. Scale and padding are retained to map model boxes back to
// the original image.
type YOLOImage struct {
	Pixels                    []float32
	Width, Height             int
	SourceWidth, SourceHeight int
	Scale, PadX, PadY         float32
}

// PrepareYOLOImage letterboxes img to cfg's model size using the standard
// YOLO pad colour (114), producing CHW RGB float32 pixels without third-party
// image or tensor dependencies.
func PrepareYOLOImage(img image.Image, cfg YOLOConfig) (YOLOImage, error) {
	if cfg.InputWidth <= 0 || cfg.InputHeight <= 0 {
		return YOLOImage{}, fmt.Errorf("preparing YOLO image: input dimensions must be positive, got %dx%d", cfg.InputWidth, cfg.InputHeight)
	}
	b := img.Bounds()
	sw, sh := b.Dx(), b.Dy()
	if sw <= 0 || sh <= 0 {
		return YOLOImage{}, fmt.Errorf("preparing YOLO image: empty image (%dx%d)", sw, sh)
	}
	scale := min(float32(cfg.InputWidth)/float32(sw), float32(cfg.InputHeight)/float32(sh))
	rw, rh := max(1, int(math.Round(float64(float32(sw)*scale)))), max(1, int(math.Round(float64(float32(sh)*scale))))
	padX, padY := (cfg.InputWidth-rw)/2, (cfg.InputHeight-rh)/2
	plane := cfg.InputWidth * cfg.InputHeight
	out := YOLOImage{Pixels: make([]float32, 3*plane), Width: cfg.InputWidth, Height: cfg.InputHeight, SourceWidth: sw, SourceHeight: sh, Scale: scale, PadX: float32(padX), PadY: float32(padY)}
	for i := range out.Pixels {
		out.Pixels[i] = 114.0 / 255.0
	}
	for y := 0; y < rh; y++ {
		sy := min(sh-1, int(float32(y)/scale))
		for x := 0; x < rw; x++ {
			sx := min(sw-1, int(float32(x)/scale))
			r, g, bl, _ := img.At(b.Min.X+sx, b.Min.Y+sy).RGBA()
			dst := (padY+y)*cfg.InputWidth + padX + x
			out.Pixels[dst] = float32(r>>8) / 255
			out.Pixels[plane+dst] = float32(g>>8) / 255
			out.Pixels[2*plane+dst] = float32(bl>>8) / 255
		}
	}
	return out, nil
}

// BoundingBox uses source-image pixel coordinates, with Max values exclusive.
type BoundingBox struct{ XMin, YMin, XMax, YMax float32 }

// Detection is one class-specific, non-max-suppressed result.
type Detection struct {
	ClassID    int
	Label      string
	Confidence float32
	Box        BoundingBox
}

// DecodeYOLOv8 converts a standard YOLOv8 detection head into source-image
// detections. output has either [4+classes, anchors] or [anchors, 4+classes]
// shape; values are cx, cy, width, height followed by per-class confidence.
// The head must already have decoded its distribution-focal-loss box channels,
// as is the case for normal exported YOLOv8 detect outputs.
func DecodeYOLOv8(output []float32, rows, cols int, input YOLOImage, cfg YOLOConfig) ([]Detection, error) {
	if rows <= 0 || cols <= 0 || rows*cols != len(output) {
		return nil, fmt.Errorf("decoding YOLOv8: output shape [%d,%d] does not match %d values", rows, cols, len(output))
	}
	if input.Width <= 0 || input.Height <= 0 || input.SourceWidth <= 0 || input.SourceHeight <= 0 || input.Scale <= 0 {
		return nil, fmt.Errorf("decoding YOLOv8: invalid prepared image")
	}
	if cfg.ScoreThreshold <= 0 {
		cfg.ScoreThreshold = 0.25
	}
	if cfg.IoUThreshold <= 0 {
		cfg.IoUThreshold = 0.45
	}
	if cfg.MaxDetections <= 0 {
		cfg.MaxDetections = 300
	}

	features, anchors, transposed := rows, cols, false
	// Normal exports are [features, anchors] (for example [84,8400]);
	// transposed exports are [anchors, features]. A one-anchor fixture also
	// needs the latter rule even though its row count is not greater than cols.
	if cols >= 5 && (rows < 5 || rows > cols) {
		features, anchors, transposed = cols, rows, true
	}
	if features < 5 {
		return nil, fmt.Errorf("decoding YOLOv8: need at least 5 features (4 box + 1 class), got %d", features)
	}
	value := func(a, f int) float32 {
		if transposed {
			return output[a*features+f]
		}
		return output[f*anchors+a]
	}
	candidates := make([]Detection, 0)
	for a := 0; a < anchors; a++ {
		class, score := 0, value(a, 4)
		for c := 1; c < features-4; c++ {
			if v := value(a, 4+c); v > score {
				class, score = c, v
			}
		}
		if score < cfg.ScoreThreshold {
			continue
		}
		cx, cy, w, h := value(a, 0), value(a, 1), value(a, 2), value(a, 3)
		if w <= 0 || h <= 0 || math.IsNaN(float64(score)) {
			continue
		}
		box := BoundingBox{(cx - w/2 - input.PadX) / input.Scale, (cy - h/2 - input.PadY) / input.Scale, (cx + w/2 - input.PadX) / input.Scale, (cy + h/2 - input.PadY) / input.Scale}
		box.XMin, box.YMin = max(float32(0), box.XMin), max(float32(0), box.YMin)
		box.XMax, box.YMax = min(float32(input.SourceWidth), box.XMax), min(float32(input.SourceHeight), box.YMax)
		if box.XMax <= box.XMin || box.YMax <= box.YMin {
			continue
		}
		label := ""
		if class < len(cfg.ClassNames) {
			label = cfg.ClassNames[class]
		}
		candidates = append(candidates, Detection{ClassID: class, Label: label, Confidence: score, Box: box})
	}
	return suppressYOLO(candidates, cfg.IoUThreshold, cfg.MaxDetections), nil
}

func suppressYOLO(candidates []Detection, threshold float32, maximum int) []Detection {
	sort.SliceStable(candidates, func(i, j int) bool { return candidates[i].Confidence > candidates[j].Confidence })
	kept := make([]Detection, 0, min(len(candidates), maximum))
	for _, candidate := range candidates {
		if len(kept) == maximum {
			break
		}
		discard := false
		for _, existing := range kept {
			if existing.ClassID == candidate.ClassID && yoloIoU(existing.Box, candidate.Box) > threshold {
				discard = true
				break
			}
		}
		if !discard {
			kept = append(kept, candidate)
		}
	}
	return kept
}

func yoloIoU(a, b BoundingBox) float32 {
	w := max(float32(0), min(a.XMax, b.XMax)-max(a.XMin, b.XMin))
	h := max(float32(0), min(a.YMax, b.YMax)-max(a.YMin, b.YMin))
	intersection := w * h
	union := (a.XMax-a.XMin)*(a.YMax-a.YMin) + (b.XMax-b.XMin)*(b.YMax-b.YMin) - intersection
	if union <= 0 {
		return 0
	}
	return intersection / union
}
