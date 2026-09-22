package server

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/SimonWaldherr/GopherLLM/internal/yolotest"
)

func visionFixture(t *testing.T) (*http.ServeMux, string) {
	t.Helper()
	// Keep stock-model resolution away from the developer's real HF cache.
	t.Setenv("HF_HUB_CACHE", t.TempDir())
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "yolo"), 0o755); err != nil {
		t.Fatal(err)
	}
	ckpt := yolotest.V8().Safetensors(func(name string) string { return "model." + name })
	if err := os.WriteFile(filepath.Join(dir, "yolo", "tiny.safetensors"), ckpt, 0o644); err != nil {
		t.Fatal(err)
	}
	// A safetensors file that is not a detector must be neither listed nor loadable.
	header := []byte(`{"lm_head.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`)
	other := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	other = append(append(other, header...), 0, 0, 0, 0)
	if err := os.WriteFile(filepath.Join(dir, "llm.safetensors"), other, 0o644); err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	registerVisionRoutes(mux, make(chan struct{}, 2), HandlerOptions{ModelDir: dir, Features: Features{ModelCatalog: true}}, &yoloModelCache{})
	return mux, dir
}

func detectionRequest(t *testing.T, fields map[string]string, img []byte) *http.Request {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	for k, v := range fields {
		if err := mw.WriteField(k, v); err != nil {
			t.Fatal(err)
		}
	}
	if img != nil {
		fw, err := mw.CreateFormFile("file", "frame.png")
		if err != nil {
			t.Fatal(err)
		}
		_, _ = fw.Write(img)
	}
	if err := mw.Close(); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/vision/detections", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	return req
}

func testPNG(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.SetRGBA(x, y, color.RGBA{R: uint8(3 * x), G: uint8(5 * y), B: uint8(x ^ y), A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestVisionRoutesListAndDetect(t *testing.T) {
	mux, _ := visionFixture(t)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/models/detection", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("listing: %d %s", rec.Code, rec.Body)
	}
	var listing struct {
		Models []detectionModel `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &listing); err != nil {
		t.Fatal(err)
	}
	ids := map[string]detectionModel{}
	for _, m := range listing.Models {
		ids[m.ID] = m
	}
	if m, ok := ids["yolo/tiny.safetensors"]; !ok || m.Source != "local" || !m.Available {
		t.Fatalf("local checkpoint missing from listing: %+v", listing.Models)
	}
	if _, ok := ids["llm.safetensors"]; ok {
		t.Fatal("a non-YOLO safetensors file was listed")
	}
	if m, ok := ids["yolov8n"]; !ok || m.Source != "stock" || m.Available || m.DownloadFile != "yolov8n.safetensors" {
		t.Fatalf("stock entry = %+v, want an unavailable yolov8n with download info", m)
	}

	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, detectionRequest(t, map[string]string{"model": "yolo/tiny.safetensors", "conf": "0.01", "size": "64"}, testPNG(t, 90, 60)))
	if rec.Code != http.StatusOK {
		t.Fatalf("detect: %d %s", rec.Code, rec.Body)
	}
	var result struct {
		Model, Version string
		Width, Height  int
		Detections     []struct {
			ClassID    int     `json:"class_id"`
			Confidence float32 `json:"confidence"`
			Box        struct {
				XMin float32 `json:"x_min"`
				XMax float32 `json:"x_max"`
			} `json:"box"`
		}
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Version != "yolov8" || result.Width != 90 || result.Height != 60 || len(result.Detections) == 0 {
		t.Fatalf("result = %+v", result)
	}
	for _, d := range result.Detections {
		if d.Box.XMin < 0 || d.Box.XMax > 90 || d.Confidence < 0.01 {
			t.Fatalf("detection outside the image or below threshold: %+v", d)
		}
	}
}

func TestVisionRoutesRejectBadRequests(t *testing.T) {
	mux, dir := visionFixture(t)
	outside := filepath.Join(t.TempDir(), "outside.safetensors")
	checkpoint, err := os.ReadFile(filepath.Join(dir, "yolo", "tiny.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(outside, checkpoint, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, "outside.safetensors")); err != nil {
		t.Fatal(err)
	}
	img := testPNG(t, 8, 8)
	for _, tc := range []struct {
		name   string
		fields map[string]string
		img    []byte
		want   int
	}{
		{"no model", map[string]string{}, img, http.StatusBadRequest},
		{"path escape", map[string]string{"model": "../tiny.safetensors"}, img, http.StatusBadRequest},
		{"absolute path", map[string]string{"model": "/etc/passwd.pt"}, img, http.StatusBadRequest},
		{"symlink escape", map[string]string{"model": "outside.safetensors"}, img, http.StatusBadRequest},
		{"not a detector", map[string]string{"model": "llm.safetensors"}, img, http.StatusBadRequest},
		{"stock not downloaded", map[string]string{"model": "yolov8n"}, img, http.StatusConflict},
		{"bad conf", map[string]string{"model": "yolo/tiny.safetensors", "conf": "2"}, img, http.StatusBadRequest},
		{"bad size", map[string]string{"model": "yolo/tiny.safetensors", "size": "100"}, img, http.StatusBadRequest},
		{"no image", map[string]string{"model": "yolo/tiny.safetensors"}, nil, http.StatusBadRequest},
		{"not an image", map[string]string{"model": "yolo/tiny.safetensors"}, []byte("hello"), http.StatusBadRequest},
		{"oversize image", map[string]string{"model": "yolo/tiny.safetensors"}, make([]byte, detectionMaxBytes+1), http.StatusRequestEntityTooLarge},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			mux.ServeHTTP(rec, detectionRequest(t, tc.fields, tc.img))
			if rec.Code != tc.want {
				t.Fatalf("status %d (%s), want %d", rec.Code, bytes.TrimSpace(rec.Body.Bytes()), tc.want)
			}
		})
	}
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/models/detection", nil))
	if rec.Code != http.StatusOK || bytes.Contains(rec.Body.Bytes(), []byte(`"id":"outside.safetensors"`)) {
		t.Fatalf("catalog exposed symlink outside model directory: %d %s", rec.Code, rec.Body.String())
	}
	if !browserDisabledPath("/v1/vision/detections") || !browserDisabledPath("/models/detection") {
		t.Fatal("vision routes must be disabled in browser deployment mode")
	}
}
