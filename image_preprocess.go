package gopherllm

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
)

// DecodeImageBytes decodes a PNG/JPEG image from raw bytes using the
// standard library's format-sniffing image.Decode (the blank image/jpeg and
// image/png imports above register those two decoders; no third-party code
// or go.mod entry is involved, both packages ship with the Go toolchain).
func DecodeImageBytes(data []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decoding image: %w", err)
	}
	return img, nil
}

// PreprocessedImage holds one image's patch grid, ready for
// PixtralVisionWeights' patch-embedding matvec. Pixels[patchIndex] is a
// flattened [channel][kh][kw] (channel slowest-varying, kw fastest) float32
// vector of length 3*PatchSize*PatchSize, already per-channel normalized —
// this ordering matches the patch_embd weight's column layout established
// in pixtral_vision.go (GGUF stores a stride==kernel conv2d weight in the
// same byte order as an ordinary linear layer over a flattened patch).
type PreprocessedImage struct {
	Pixels     [][]float32
	Rows, Cols int // patch grid shape (Rows*Cols == len(Pixels))
}

// PreprocessImagePixtral resizes img so its longest edge is at most maxEdge
// (preserving aspect ratio, never forcing a square), rounds each resulting
// dimension up to the next multiple of roundTo (typically patchSize, or
// patchSize*spatialMergeSize when the vision tower merges adjacent patches
// after encoding, so the raw patch grid divides evenly by the merge size
// too) so the patch grid divides evenly, then extracts and normalizes
// non-overlapping patchSize x patchSize patches in row-major (row, then
// column) order.
func PreprocessImagePixtral(img image.Image, patchSize, roundTo, maxEdge int, mean, std [3]float32) (*PreprocessedImage, error) {
	if patchSize <= 0 {
		return nil, fmt.Errorf("preprocessing image: patch size must be positive, got %d", patchSize)
	}
	if roundTo <= 0 {
		roundTo = patchSize
	}
	b := img.Bounds()
	srcW, srcH := b.Dx(), b.Dy()
	if srcW <= 0 || srcH <= 0 {
		return nil, fmt.Errorf("preprocessing image: empty image (%dx%d)", srcW, srcH)
	}

	dstW, dstH := srcW, srcH
	if longest := max(dstW, dstH); maxEdge > 0 && longest > maxEdge {
		scale := float64(maxEdge) / float64(longest)
		dstW = max(1, int(float64(dstW)*scale))
		dstH = max(1, int(float64(dstH)*scale))
	}
	dstW = roundUpToMultiple(dstW, roundTo)
	dstH = roundUpToMultiple(dstH, roundTo)

	pixels := resizeBilinearNormalized(img, dstW, dstH, mean, std)

	cols := dstW / patchSize
	rows := dstH / patchSize
	patchLen := 3 * patchSize * patchSize
	out := &PreprocessedImage{Pixels: make([][]float32, 0, rows*cols), Rows: rows, Cols: cols}
	for py := 0; py < rows; py++ {
		for px := 0; px < cols; px++ {
			patch := make([]float32, patchLen)
			for c := 0; c < 3; c++ {
				chanBase := c * patchSize * patchSize
				for kh := 0; kh < patchSize; kh++ {
					srcY := py*patchSize + kh
					rowBase := chanBase + kh*patchSize
					for kw := 0; kw < patchSize; kw++ {
						srcX := px*patchSize + kw
						patch[rowBase+kw] = pixels[(srcY*dstW+srcX)*3+c]
					}
				}
			}
			out.Pixels = append(out.Pixels, patch)
		}
	}
	return out, nil
}

// PreprocessImagePixtralDynamic implements the dynamic Pixtral sizing rule
// used by Ministral/Mistral3: dimensions are rounded to the nearest post-merge
// patch multiple, then constrained by a minimum/maximum number of image tokens
// (one token corresponds to align*align pixels). This intentionally differs
// from a simple "round up" rule: a 384x201 webcam frame becomes 392x196, not
// 392x224, preserving the visual-token grid used during training.
func PreprocessImagePixtralDynamic(img image.Image, patchSize, mergeSize, minTokens, maxTokens int, mean, std [3]float32) (*PreprocessedImage, error) {
	if patchSize <= 0 {
		return nil, fmt.Errorf("preprocessing image: patch size must be positive, got %d", patchSize)
	}
	mergeSize = max(1, mergeSize)
	align := patchSize * mergeSize
	if minTokens <= 0 {
		minTokens = 8
	}
	if maxTokens < minTokens {
		maxTokens = max(minTokens, 1024)
	}
	b := img.Bounds()
	srcW, srcH := b.Dx(), b.Dy()
	if srcW <= 0 || srcH <= 0 {
		return nil, fmt.Errorf("preprocessing image: empty image (%dx%d)", srcW, srcH)
	}
	dstW, dstH := pixtralDynamicImageSize(srcW, srcH, align, minTokens*align*align, maxTokens*align*align)
	pixels := resizeBilinearNormalized(img, dstW, dstH, mean, std)

	cols := dstW / patchSize
	rows := dstH / patchSize
	patchLen := 3 * patchSize * patchSize
	out := &PreprocessedImage{Pixels: make([][]float32, 0, rows*cols), Rows: rows, Cols: cols}
	for py := 0; py < rows; py++ {
		for px := 0; px < cols; px++ {
			patch := make([]float32, patchLen)
			for c := 0; c < 3; c++ {
				chanBase := c * patchSize * patchSize
				for kh := 0; kh < patchSize; kh++ {
					srcY := py*patchSize + kh
					rowBase := chanBase + kh*patchSize
					for kw := 0; kw < patchSize; kw++ {
						srcX := px*patchSize + kw
						patch[rowBase+kw] = pixels[(srcY*dstW+srcX)*3+c]
					}
				}
			}
			out.Pixels = append(out.Pixels, patch)
		}
	}
	return out, nil
}

// pixtralDynamicImageSize mirrors Pixtral's smart resize calculation. It
// keeps the final dimensions aligned while minimizing aspect distortion under
// the image-token bounds. The caller validates source dimensions and align.
func pixtralDynamicImageSize(srcW, srcH, align, minPixels, maxPixels int) (dstW, dstH int) {
	factor := float32(align)
	roundAligned := func(v int) int {
		// mtmd performs this computation in float before std::round. Retaining
		// that precision avoids rare one-grid-cell disagreements at boundaries.
		return max(align, int(math.Round(float64(float32(v)/factor)))*align)
	}
	floorAligned := func(v float32) int {
		return max(align, int(math.Floor(float64(v/factor)))*align)
	}
	ceilAligned := func(v float32) int {
		return max(align, int(math.Ceil(float64(v/factor)))*align)
	}
	dstW, dstH = roundAligned(srcW), roundAligned(srcH)
	if dstW*dstH > maxPixels {
		beta := float32(math.Sqrt(float64(float32(srcW*srcH) / float32(maxPixels))))
		dstW, dstH = floorAligned(float32(srcW)/beta), floorAligned(float32(srcH)/beta)
	} else if dstW*dstH < minPixels {
		beta := float32(math.Sqrt(float64(float32(minPixels) / float32(srcW*srcH))))
		dstW, dstH = ceilAligned(float32(srcW)*beta), ceilAligned(float32(srcH)*beta)
	}
	return dstW, dstH
}

func roundUpToMultiple(v, m int) int {
	if m <= 0 {
		return v
	}
	return ((v + m - 1) / m) * m
}

// resizeBilinearNormalized resizes img to dstW x dstH using Pixtral's
// reference bilinear sampling: source and destination corners align, and each
// interpolated RGB value is truncated back to uint8 before normalization.
//
// That latter detail matters. The companion projector was trained on the
// reference preprocessor's 8-bit resize output; leaving interpolation values
// as floats changes the patch-convolution input enough for encoder activations
// to diverge. The result is row-major, interleaved RGB float32 values already
// normalized via (v-mean[c])/std[c].
func resizeBilinearNormalized(img image.Image, dstW, dstH int, mean, std [3]float32) []float32 {
	b := img.Bounds()
	srcW, srcH := b.Dx(), b.Dy()
	out := make([]float32, dstW*dstH*3)

	sampleAt := func(x, y int) (r, g, bl uint8) {
		rr, gg, bb, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
		// image.Image exposes channels in 16-bit form. The mtmd reference path
		// consumes an 8-bit RGB bitmap, so keep the corresponding high byte
		// before resize rather than interpolating 16-bit/float source values.
		return uint8(rr >> 8), uint8(gg >> 8), uint8(bb >> 8)
	}

	// Match mtmd's resize_bilinear exactly: its endpoints are aligned rather
	// than sampled at pixel centers. The zero-sized ratio handles a 1px output.
	scaleX, scaleY := float32(0), float32(0)
	if dstW > 1 {
		scaleX = float32(srcW-1) / float32(dstW-1)
	}
	if dstH > 1 {
		scaleY = float32(srcH-1) / float32(dstH-1)
	}
	for dy := 0; dy < dstH; dy++ {
		sy := float32(dy) * scaleY
		y0 := int(sy)
		y1 := min(y0+1, srcH-1)
		fy := sy - float32(y0)
		for dx := 0; dx < dstW; dx++ {
			sx := float32(dx) * scaleX
			x0 := int(sx)
			x1 := min(x0+1, srcW-1)
			fx := sx - float32(x0)

			r00, g00, b00 := sampleAt(x0, y0)
			r10, g10, b10 := sampleAt(x1, y0)
			r01, g01, b01 := sampleAt(x0, y1)
			r11, g11, b11 := sampleAt(x1, y1)

			r := bilerpU8(r00, r10, r01, r11, fx, fy)
			g := bilerpU8(g00, g10, g01, g11, fx, fy)
			bc := bilerpU8(b00, b10, b01, b11, fx, fy)

			idx := (dy*dstW + dx) * 3
			out[idx+0] = (float32(r)/255 - mean[0]) / std[0]
			out[idx+1] = (float32(g)/255 - mean[1]) / std[1]
			out[idx+2] = (float32(bc)/255 - mean[2]) / std[2]
		}
	}
	return out
}

func bilerpU8(v00, v10, v01, v11 uint8, fx, fy float32) uint8 {
	top := float32(v00) + (float32(v10)-float32(v00))*fx
	bottom := float32(v01) + (float32(v11)-float32(v01))*fx
	return uint8(top + (bottom-top)*fy)
}
