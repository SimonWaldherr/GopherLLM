// Command yolo-detect runs GopherLLM's native YOLO object detector (YOLOv8
// and YOLO11; see yolo.go and yolo_model.go in the module root) over one
// image.
//
// GopherLLM does not ship YOLO weights. --model accepts a stock name, which
// is downloaded once from Hugging Face into the standard HF cache and reused
// offline afterwards, an explicit hf: reference, or a local file:
//
//	go run ./cmd/yolo-detect --model yolo11n --image street.jpg
//	go run ./cmd/yolo-detect --model hf:owner/repo:best.safetensors --image x.png --labels labels.txt
//	go run ./cmd/yolo-detect --model ./best.safetensors --image x.jpg --out boxes.png
//
// Ultralytics .pt checkpoints (yolo11n.pt, a custom-trained best.pt, ...)
// load directly. Their pickle is read as data only; nothing in the file is
// executed, and no Python is needed:
//
//	go run ./cmd/yolo-detect --model best.pt --image x.jpg
//
// The stock weights are Ultralytics' and licensed AGPL-3.0.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	"io"
	"os"
	"strings"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/huggingface"
)

func main() {
	modelRef := flag.String("model", "yolo11n", "yolo11n|s|m|l|x or yolov8n|s|m|l|x (downloaded from Hugging Face on first use), hf:owner/repo:file[@rev], or a local .safetensors/.pt path")
	imagePath := flag.String("image", "", "path to a JPEG or PNG image")
	size := flag.Int("size", 640, "square network input size, a multiple of 32")
	conf := flag.Float64("conf", 0.25, "minimum class confidence")
	iou := flag.Float64("iou", 0.45, "IoU threshold for non-max suppression")
	labelsPath := flag.String("labels", "", "file with one class name per line (default: the checkpoint's own names, else COCO names for 80-class models)")
	outPath := flag.String("out", "", "write a PNG copy of the image with the detections drawn in")
	asJSON := flag.Bool("json", false, "print detections as JSON")
	offline := flag.Bool("offline", false, "never contact Hugging Face; use only the local cache")
	verbose := flag.Bool("v", false, "print download progress and timings to stderr")
	flag.Parse()

	if *imagePath == "" {
		fmt.Fprintln(os.Stderr, "usage: yolo-detect [--model yolo11n] --image photo.jpg [--out boxes.png] [--json]")
		os.Exit(2)
	}
	if err := run(*modelRef, *imagePath, *size, float32(*conf), float32(*iou), *labelsPath, *outPath, *asJSON, *offline, *verbose); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(modelRef, imagePath string, size int, conf, iou float32, labelsPath, outPath string, asJSON, offline, verbose bool) error {
	logw := io.Discard
	if verbose {
		logw = os.Stderr
	}
	path, err := resolveModel(modelRef, offline, logw)
	if err != nil {
		return err
	}
	start := time.Now()
	model, err := gopherllm.LoadYOLO(path)
	if err != nil {
		return err
	}
	if labelsPath != "" {
		if model.ClassNames, err = readLabels(labelsPath); err != nil {
			return err
		}
		if len(model.ClassNames) != model.NumClasses {
			fmt.Fprintf(os.Stderr, "warning: %s has %d labels, model has %d classes\n", labelsPath, len(model.ClassNames), model.NumClasses)
		}
	} else if model.ClassNames == nil && model.NumClasses == len(gopherllm.COCOClassNames) {
		model.ClassNames = gopherllm.COCOClassNames
	}
	loaded := time.Since(start)

	f, err := os.Open(imagePath)
	if err != nil {
		return err
	}
	img, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		return fmt.Errorf("decoding %s: %w", imagePath, err)
	}

	cfg := gopherllm.DefaultYOLOConfig()
	cfg.InputWidth, cfg.InputHeight, cfg.ScoreThreshold, cfg.IoUThreshold = size, size, conf, iou
	start = time.Now()
	dets, err := model.Detect(img, cfg)
	if err != nil {
		return err
	}
	if verbose {
		fmt.Fprintf(os.Stderr, "model %s (%s): %d classes, loaded in %v; detection took %v\n", path, model.Version, model.NumClasses, loaded.Round(time.Millisecond), time.Since(start).Round(time.Millisecond))
	}

	if asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(dets); err != nil {
			return err
		}
	} else {
		for _, d := range dets {
			label := d.Label
			if label == "" {
				label = fmt.Sprintf("class %d", d.ClassID)
			}
			fmt.Printf("%-16s %.2f  x=%.0f..%.0f y=%.0f..%.0f\n", label, d.Confidence, d.Box.XMin, d.Box.XMax, d.Box.YMin, d.Box.YMax)
		}
	}
	if outPath != "" {
		return writeAnnotated(img, dets, outPath)
	}
	return nil
}

// resolveModel turns --model into a local file, downloading through the
// shared Hugging Face client (and its cache) only for stock names and
// explicit hf: references.
func resolveModel(ref string, offline bool, logw io.Writer) (string, error) {
	if pinned, ok := gopherllm.YOLOCheckpointReference(ref); ok {
		if _, err := os.Stat(ref); err != nil { // a local file of that name wins
			ref = pinned
		}
	}
	if !strings.HasPrefix(strings.ToLower(ref), "hf:") {
		return ref, nil
	}
	spec, rev, hasRev := strings.Cut(ref[len("hf:"):], "@")
	repo, file, ok := strings.Cut(spec, ":")
	if !ok || file == "" {
		return "", fmt.Errorf("hf reference %q must name a file: hf:owner/repo:file.safetensors[@revision]", ref)
	}
	if hasRev {
		repo += "@" + rev
	}
	opts := huggingface.DefaultOptions()
	opts.Offline = opts.Offline || offline
	paths, err := huggingface.DownloadFiles(context.Background(), repo, []string{file}, logw, opts)
	if err != nil {
		return "", fmt.Errorf("fetching %s: %w", ref, err)
	}
	return paths[0], nil
}

func readLabels(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var labels []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if line := strings.TrimSpace(sc.Text()); line != "" {
			labels = append(labels, line)
		}
	}
	return labels, sc.Err()
}

// writeAnnotated draws each detection as a 3px outline, coloured by class.
func writeAnnotated(src image.Image, dets []gopherllm.Detection, path string) error {
	b := src.Bounds()
	img := image.NewRGBA(b)
	draw.Draw(img, b, src, b.Min, draw.Src)
	palette := []color.RGBA{{230, 25, 75, 255}, {60, 180, 75, 255}, {255, 225, 25, 255}, {0, 130, 200, 255}, {245, 130, 48, 255}, {145, 30, 180, 255}, {70, 240, 240, 255}, {240, 50, 230, 255}}
	for _, d := range dets {
		c := palette[d.ClassID%len(palette)]
		x0, y0 := b.Min.X+int(d.Box.XMin), b.Min.Y+int(d.Box.YMin)
		x1, y1 := b.Min.X+int(d.Box.XMax)-1, b.Min.Y+int(d.Box.YMax)-1
		for t := range 3 {
			for x := x0; x <= x1; x++ {
				img.SetRGBA(x, y0+t, c)
				img.SetRGBA(x, y1-t, c)
			}
			for y := y0; y <= y1; y++ {
				img.SetRGBA(x0+t, y, c)
				img.SetRGBA(x1-t, y, c)
			}
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(f, img); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
