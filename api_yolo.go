package gopherllm

import (
	"image"

	"github.com/SimonWaldherr/GopherLLM/internal/vision/yolo"
)

// YOLOConfig controls native YOLO image preparation and post-processing.
// It aliases the vision package type to preserve the root API.
type YOLOConfig = yolo.YOLOConfig

// YOLOImage is a letterboxed, CHW RGB image prepared for a detector.
type YOLOImage = yolo.YOLOImage

// BoundingBox uses source-image pixel coordinates, with Max values exclusive.
type BoundingBox = yolo.BoundingBox

// Detection is one class-specific, non-max-suppressed result.
type Detection = yolo.Detection

// YOLOModel is a loaded YOLOv8 or YOLO11 detection network.
type YOLOModel = yolo.YOLOModel

// COCOClassNames are the 80 COCO detection labels in class-index order.
var COCOClassNames = yolo.COCOClassNames

// YOLOStockModels lists stock names understood by YOLOCheckpointReference.
var YOLOStockModels = yolo.YOLOStockModels

// DefaultYOLOConfig returns defaults for a 640x640 detector input.
func DefaultYOLOConfig() YOLOConfig { return yolo.DefaultYOLOConfig() }

// PrepareYOLOImage letterboxes an image into CHW RGB float32 pixels.
func PrepareYOLOImage(img image.Image, cfg YOLOConfig) (YOLOImage, error) {
	return yolo.PrepareYOLOImage(img, cfg)
}

// DecodeYOLOv8 decodes standard or transposed YOLOv8 head output.
func DecodeYOLOv8(output []float32, rows, cols int, input YOLOImage, cfg YOLOConfig) ([]Detection, error) {
	return yolo.DecodeYOLOv8(output, rows, cols, input, cfg)
}

// LoadYOLO loads a safetensors or Ultralytics .pt detector checkpoint.
func LoadYOLO(path string) (*YOLOModel, error) { return yolo.LoadYOLO(path) }

// IsYOLOCheckpoint reports whether a file looks like a supported detector checkpoint.
func IsYOLOCheckpoint(path string) bool { return yolo.IsYOLOCheckpoint(path) }

// LoadYOLOSafetensors loads a detector from safetensors bytes.
func LoadYOLOSafetensors(data []byte) (*YOLOModel, error) {
	return yolo.LoadYOLOSafetensors(data)
}

// LoadYOLOPyTorch loads an Ultralytics .pt detector checkpoint without
// executing Python pickle code.
func LoadYOLOPyTorch(data []byte) (*YOLOModel, error) {
	return yolo.LoadYOLOPyTorch(data)
}

// YOLOCheckpointReference maps a stock name to its pinned Hugging Face file
// reference. It builds the reference but does not download the checkpoint.
func YOLOCheckpointReference(name string) (string, bool) {
	return yolo.YOLOCheckpointReference(name)
}
