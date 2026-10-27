package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func TestImageGenerationRequest(t *testing.T) {
	var calls int
	mux := http.NewServeMux()
	closeRoutes := registerImageRoutes(mux, HandlerOptions{ImageModelDir: "configured", ChatUI: true}, func(ctx context.Context, opts gopherllm.QwenImage21Options, keepAlive bool) (*image.NRGBA, error) {
		calls++
		if opts.Prompt != "teapot" || opts.Width != 512 || opts.Height != 512 || opts.Seed != 42 || opts.Steps != 0 || !keepAlive {
			t.Errorf("options = %+v", opts)
		}
		img := image.NewNRGBA(image.Rect(0, 0, 2, 2))
		img.SetNRGBA(0, 0, color.NRGBA{R: 255, A: 255})
		return img, nil
	}, nil)
	defer closeRoutes()
	for _, body := range []string{`{}`, `null`, `{"prompt":" "}`, `{"prompt":"x","size":"2048x2048"}`, `{"prompt":"x","n":2}`, `{"prompt":"x","response_format":"url"}`, `{"prompt":"x","model":"/tmp/private"}`, `{"prompt":"x"} {}`, `{"prompt":"` + strings.Repeat("x", 4097) + `"}`, strings.Repeat(" ", 32<<10) + `{}`} {
		res := httptest.NewRecorder()
		mux.ServeHTTP(res, httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(body)))
		if res.Code != 400 {
			t.Errorf("invalid request status=%d", res.Code)
		}
	}
	if calls != 0 {
		t.Fatal("invalid request loaded model")
	}
	res := httptest.NewRecorder()
	mux.ServeHTTP(res, httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"prompt":" teapot ","size":"512x512","seed":42,"keep_alive":true,"n":1,"response_format":"b64_json"}`)))
	if res.Code != 200 {
		t.Fatalf("response: %d %s", res.Code, res.Body.String())
	}
	var result struct {
		Data []struct {
			B64 string `json:"b64_json"`
		}
	}
	if err := json.Unmarshal(res.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Data) != 1 {
		t.Fatalf("data=%+v", result.Data)
	}
	data, err := base64.StdEncoding.DecodeString(result.Data[0].B64)
	if err != nil {
		t.Fatal(err)
	}
	img, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	r, g, b, a := img.At(0, 0).RGBA()
	if r != 65535 || g != 0 || b != 0 || a != 65535 {
		t.Fatal("PNG pixel changed")
	}
	if calls != 1 || res.Header().Get("Cache-Control") != "no-store" {
		t.Fatal("generation/cache contract")
	}
	res = httptest.NewRecorder()
	mux.ServeHTTP(res, httptest.NewRequest("GET", "/v1/images/generations", nil))
	if res.Code != 405 {
		t.Fatalf("GET=%d", res.Code)
	}
}

func TestImageGenerationCancellationAndClose(t *testing.T) {
	for _, shutdown := range []bool{false, true} {
		t.Run(map[bool]string{false: "client", true: "shutdown"}[shutdown], func(t *testing.T) {
			entered := make(chan struct{})
			var released atomic.Bool
			mux := http.NewServeMux()
			closeRoutes := registerImageRoutes(mux, HandlerOptions{ImageModelDir: "configured"}, func(ctx context.Context, _ gopherllm.QwenImage21Options, _ bool) (*image.NRGBA, error) {
				close(entered)
				<-ctx.Done()
				released.Store(true)
				return nil, ctx.Err()
			}, nil)
			defer closeRoutes()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan struct{})
			res := httptest.NewRecorder()
			go func() {
				defer close(done)
				mux.ServeHTTP(res, httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"prompt":"x"}`)).WithContext(ctx))
			}()
			select {
			case <-entered:
			case <-time.After(5 * time.Second):
				t.Fatal("generation did not start")
			}
			rejected := httptest.NewRecorder()
			mux.ServeHTTP(rejected, httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"prompt":"y"}`)))
			if rejected.Code != 429 {
				t.Fatalf("concurrent status=%d", rejected.Code)
			}
			unload := httptest.NewRecorder()
			mux.ServeHTTP(unload, httptest.NewRequest("POST", "/v1/images/unload", nil))
			if unload.Code != 429 {
				t.Fatalf("unload during generation = %d", unload.Code)
			}
			if shutdown {
				closeRoutes()
				if !released.Load() {
					t.Fatal("close returned before release")
				}
			} else {
				cancel()
			}
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				t.Fatal("generation did not stop")
			}
			if res.Code == 200 {
				t.Fatal("canceled generation succeeded")
			}
			closeRoutes()
			rejected = httptest.NewRecorder()
			mux.ServeHTTP(rejected, httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"prompt":"z"}`)))
			if rejected.Code != 503 {
				t.Fatalf("closed status=%d", rejected.Code)
			}
		})
	}
}

func TestImageGenerationDeployment(t *testing.T) {
	for _, tc := range []struct {
		name      string
		opts      HandlerOptions
		available bool
	}{
		{"disabled", HandlerOptions{ChatUI: true}, false},
		{"local", HandlerOptions{ChatUI: true, ImageModelDir: "configured"}, true},
		{"managed", HandlerOptions{ChatUI: true, ImageModelDir: "configured", DeploymentMode: DeploymentManaged, AdminToken: "secret"}, true},
		{"browser", HandlerOptions{ChatUI: true, ImageModelDir: "configured", DeploymentMode: DeploymentBrowser}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := NewHandler(nil, tc.opts)
			defer h.Close()
			for _, path := range []string{"/images", "/images.js", "/images.css"} {
				res := httptest.NewRecorder()
				h.ServeHTTP(res, httptest.NewRequest("GET", path, nil))
				if (res.Code == 200) != tc.available {
					t.Fatalf("%s status=%d", path, res.Code)
				}
			}
			res := httptest.NewRecorder()
			h.ServeHTTP(res, httptest.NewRequest("GET", "/chat", nil))
			if strings.Contains(res.Body.String(), `href="/images"`) != tc.available {
				t.Fatal("chat link does not match availability")
			}
			res = httptest.NewRecorder()
			h.ServeHTTP(res, httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{}`)))
			if tc.available && res.Code != 400 || !tc.available && res.Code == 400 {
				t.Fatalf("API policy=%d", res.Code)
			}
		})
	}
}

func TestImageGenerationDeadline(t *testing.T) {
	for _, timeout := range []time.Duration{0, 25 * time.Millisecond} {
		observed := time.Duration(0)
		handler := withInferenceDeadline(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			deadline, ok := r.Context().Deadline()
			if !ok {
				t.Fatal("missing deadline")
			}
			observed = time.Until(deadline)
		}), timeout)
		handler.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/v1/images/generations", nil))
		want := timeout
		if want == 0 {
			want = 10 * time.Minute
		}
		if observed <= 0 || observed > want || observed < want-time.Second {
			t.Fatalf("deadline=%s want %s", observed, want)
		}
	}
}

func TestImageGenerationErrorsDoNotExposePaths(t *testing.T) {
	mux := http.NewServeMux()
	var log bytes.Buffer
	closeRoutes := registerImageRoutes(mux, HandlerOptions{ImageModelDir: "configured", LogWriter: &log}, func(context.Context, gopherllm.QwenImage21Options, bool) (*image.NRGBA, error) {
		return nil, errors.New("open /private/model/config.json: missing")
	}, nil)
	defer closeRoutes()
	res := httptest.NewRecorder()
	mux.ServeHTTP(res, httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"prompt":"x"}`)))
	if res.Code != 500 || strings.Contains(res.Body.String(), "/private/") || !strings.Contains(log.String(), "/private/") {
		t.Fatal("loader error handling")
	}
}

type testImageModel struct {
	calls, closes int
	err           error
	cancel        context.CancelFunc
}

func (m *testImageModel) Generate(context.Context, gopherllm.QwenImage21Options) (*image.NRGBA, error) {
	m.calls++
	if m.cancel != nil {
		m.cancel()
	}
	return image.NewNRGBA(image.Rect(0, 0, 2, 2)), m.err
}
func (m *testImageModel) Close() error { m.closes++; return nil }

func TestImageModelRetention(t *testing.T) {
	var opened []*testImageModel
	cache := &imageModelCache{open: func(context.Context) (imageGenerationModel, error) {
		m := &testImageModel{}
		opened = append(opened, m)
		return m, nil
	}}
	generate := func(keep bool) {
		t.Helper()
		if _, err := cache.generate(context.Background(), gopherllm.QwenImage21Options{}, keep); err != nil {
			t.Fatal(err)
		}
	}
	generate(true)
	generate(true)
	if len(opened) != 1 || opened[0].calls != 2 || opened[0].closes != 0 {
		t.Fatal("retained model was not reused")
	}
	generate(false)
	if len(opened) != 1 || opened[0].closes != 1 || cache.model != nil {
		t.Fatal("release request did not release retained model")
	}
	generate(false)
	if len(opened) != 2 || opened[1].closes != 1 {
		t.Fatal("default generation retained model")
	}
	generate(true)
	cache.close()
	cache.close()
	if opened[2].closes != 1 {
		t.Fatal("explicit close must be idempotent")
	}
	for _, cancelled := range []bool{false, true} {
		m := &testImageModel{}
		ctx, cancel := context.WithCancel(context.Background())
		if cancelled {
			m.cancel = cancel
		} else {
			m.err = errors.New("generation failed")
		}
		cache.model = m
		_, _ = cache.generate(ctx, gopherllm.QwenImage21Options{}, true)
		cancel()
		if m.closes != 1 || cache.model != nil {
			t.Fatal("failed/cancelled generation retained model")
		}
	}
	attempts := 0
	cache.open = func(context.Context) (imageGenerationModel, error) {
		attempts++
		var m *testImageModel
		return m, errors.New("load failed")
	}
	for i := 0; i < 2; i++ {
		if _, err := cache.generate(context.Background(), gopherllm.QwenImage21Options{}, true); err == nil {
			t.Fatal("missing load error")
		}
	}
	if attempts != 2 || cache.model != nil {
		t.Fatal("failed load poisoned cache")
	}
}

func TestImageUnloadAndShutdown(t *testing.T) {
	model := &testImageModel{}
	cache := &imageModelCache{open: func(context.Context) (imageGenerationModel, error) { return model, nil }}
	mux := http.NewServeMux()
	closeRoutes := registerImageRoutes(mux, HandlerOptions{ImageModelDir: "configured"}, cache.generate, cache.close)
	defer closeRoutes()
	request := func(path, body string) int {
		res := httptest.NewRecorder()
		mux.ServeHTTP(res, httptest.NewRequest("POST", path, strings.NewReader(body)))
		return res.Code
	}
	if request("/v1/images/generations", `{"prompt":"x","keep_alive":true}`) != 200 || model.closes != 0 {
		t.Fatal("keep_alive did not reach model cache")
	}
	if request("/v1/images/unload", "") != 200 || model.closes != 1 {
		t.Fatal("unload did not close cached model")
	}
	if request("/v1/images/unload", "") != 200 || model.closes != 1 {
		t.Fatal("unload must be idempotent")
	}
	if request("/v1/images/generations", `{"prompt":"x","keep_alive":true}`) != 200 {
		t.Fatal("could not reload")
	}
	closeRoutes()
	if model.closes != 2 {
		t.Fatal("shutdown retained model")
	}
	if request("/v1/images/unload", "") != 503 {
		t.Fatal("closed route accepted request")
	}
}
