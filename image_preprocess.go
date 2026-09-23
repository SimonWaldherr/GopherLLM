package gopherllm

import (
	"image"

	visionimage "github.com/SimonWaldherr/GopherLLM/internal/vision/image"
)

// DecodeImageBytes decodes an image from its encoded bytes.
func DecodeImageBytes(data []byte) (image.Image, error) {
	return visionimage.DecodeImageBytes(data)
}

// PreprocessedImage holds a patch grid ready for Pixtral's vision encoder.
type PreprocessedImage = visionimage.PreprocessedImage

// PreprocessImagePixtral resizes and normalizes an image into fixed-size patches.
func PreprocessImagePixtral(img image.Image, patchSize, roundTo, maxEdge int, mean, std [3]float32) (*PreprocessedImage, error) {
	return visionimage.PreprocessImagePixtral(img, patchSize, roundTo, maxEdge, mean, std)
}

// PreprocessImagePixtralDynamic applies Pixtral's dynamic image-sizing rule.
func PreprocessImagePixtralDynamic(img image.Image, patchSize, mergeSize, minTokens, maxTokens int, mean, std [3]float32) (*PreprocessedImage, error) {
	return visionimage.PreprocessImagePixtralDynamic(img, patchSize, mergeSize, minTokens, maxTokens, mean, std)
}
