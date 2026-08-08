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

func roundUpToMultiple(v, m int) int {
	if m <= 0 {
		return v
	}
	return ((v + m - 1) / m) * m
}

// resizeBilinearNormalized resizes img to dstW x dstH with hand-written
// bilinear interpolation (pixel-center sampling, clamped at the border) and
// returns row-major interleaved RGB float32 values, each channel already
// rescaled to [0,1] and normalized via (v-mean[c])/std[c].
func resizeBilinearNormalized(img image.Image, dstW, dstH int, mean, std [3]float32) []float32 {
	b := img.Bounds()
	srcW, srcH := b.Dx(), b.Dy()
	out := make([]float32, dstW*dstH*3)

	sampleAt := func(x, y int) (r, g, bl float32) {
		if x < 0 {
			x = 0
		} else if x >= srcW {
			x = srcW - 1
		}
		if y < 0 {
			y = 0
		} else if y >= srcH {
			y = srcH - 1
		}
		rr, gg, bb, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
		return float32(rr) / 65535, float32(gg) / 65535, float32(bb) / 65535
	}

	scaleX := float64(srcW) / float64(dstW)
	scaleY := float64(srcH) / float64(dstH)
	for dy := 0; dy < dstH; dy++ {
		sy := (float64(dy)+0.5)*scaleY - 0.5
		y0 := int(math.Floor(sy))
		fy := float32(sy - float64(y0))
		for dx := 0; dx < dstW; dx++ {
			sx := (float64(dx)+0.5)*scaleX - 0.5
			x0 := int(math.Floor(sx))
			fx := float32(sx - float64(x0))

			r00, g00, b00 := sampleAt(x0, y0)
			r10, g10, b10 := sampleAt(x0+1, y0)
			r01, g01, b01 := sampleAt(x0, y0+1)
			r11, g11, b11 := sampleAt(x0+1, y0+1)

			r := bilerp(r00, r10, r01, r11, fx, fy)
			g := bilerp(g00, g10, g01, g11, fx, fy)
			bc := bilerp(b00, b10, b01, b11, fx, fy)

			idx := (dy*dstW + dx) * 3
			out[idx+0] = (r - mean[0]) / std[0]
			out[idx+1] = (g - mean[1]) / std[1]
			out[idx+2] = (bc - mean[2]) / std[2]
		}
	}
	return out
}

func bilerp(v00, v10, v01, v11, fx, fy float32) float32 {
	top := v00 + (v10-v00)*fx
	bottom := v01 + (v11-v01)*fx
	return top + (bottom-top)*fy
}
