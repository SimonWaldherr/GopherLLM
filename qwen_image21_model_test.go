package gopherllm

import (
	"context"
	"encoding/json"
	"errors"
	"image"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

func qi21SmallOptions() QwenImage21Options {
	return QwenImage21Options{Prompt: "A blue cube.", Width: 32, Height: 32, Seed: 431}
}

func TestQwenImage21ModelDeterminismAndImageOwnership(t *testing.T) {
	m := qi21PipelineFixture(t)
	opts := qi21SmallOptions()
	a, err := m.Generate(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	want := slices.Clone(a.Pix)
	clear(a.Pix)
	b, err := m.Generate(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if b.Bounds() != image.Rect(0, 0, 32, 32) || !slices.Equal(b.Pix, want) {
		t.Fatal("identical seed did not reproduce the independently owned image")
	}
	if slices.Equal(a.Pix, b.Pix) {
		t.Fatal("generation reused caller-owned pixels")
	}
	// Equal positive/negative conditioning cancels the CFG difference exactly,
	// while still exercising two independently cached transformer branches.
	guided := opts
	guided.GuidanceScale, guided.NegativePrompt = 2, opts.Prompt
	withGuidance, err := m.Generate(context.Background(), guided)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(withGuidance.Pix, b.Pix) {
		t.Fatal("identical CFG branches changed the unguided prediction")
	}
	opts.Seed++
	c, err := m.Generate(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(b.Pix, c.Pix) {
		t.Fatal("different seeds produced identical pixels")
	}
}

func TestQwenImage21ModelDefaults(t *testing.T) {
	m := qi21PipelineFixture(t)
	steps := 0
	opts := QwenImage21Options{Prompt: "A blue cube.", Progress: func(completed, total int) {
		steps++
		if completed != steps || total != 8 {
			t.Errorf("Turbo progress %d/%d, expected %d/8", completed, total, steps)
		}
	}}
	img, err := m.Generate(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if img.Bounds() != image.Rect(0, 0, 256, 256) || steps != 8 {
		t.Fatalf("defaults produced %v with %d steps", img.Bounds(), steps)
	}
	// The same tiny components can exercise the base checkpoint's schedule;
	// only its metadata differs from Turbo for this public API contract.
	m.sigmas, m.scheduler = nil, qi21BaseScheduler()
	opts = qi21SmallOptions()
	steps = 0
	opts.Progress = func(completed, total int) {
		steps++
		if completed != steps || total != 40 {
			t.Errorf("base progress %d/%d, expected %d/40", completed, total, steps)
		}
	}
	if _, err := m.Generate(context.Background(), opts); err != nil {
		t.Fatal(err)
	}
	if steps != 40 {
		t.Fatalf("base default used %d steps", steps)
	}
}

func TestQwenImage21ModelRejectsInvalidOptions(t *testing.T) {
	m := qi21PipelineFixture(t)
	for _, tc := range []struct {
		name   string
		change func(*QwenImage21Options)
	}{
		{"empty-prompt", func(o *QwenImage21Options) { o.Prompt = "" }},
		{"blank-prompt", func(o *QwenImage21Options) { o.Prompt = " \t\n" }},
		{"long-prompt", func(o *QwenImage21Options) { o.Prompt = strings.Repeat("a", (64<<10)+1) }},
		{"long-negative", func(o *QwenImage21Options) { o.NegativePrompt = strings.Repeat("a", (64<<10)+1); o.GuidanceScale = 2 }},
		{"small-width", func(o *QwenImage21Options) { o.Width = 16 }},
		{"unaligned-width", func(o *QwenImage21Options) { o.Width = 48 }},
		{"large-height", func(o *QwenImage21Options) { o.Height = 2080 }},
		{"negative-height", func(o *QwenImage21Options) { o.Height = -32 }},
		{"negative-steps", func(o *QwenImage21Options) { o.Steps = -1 }},
		{"wrong-turbo-steps", func(o *QwenImage21Options) { o.Steps = 7 }},
		{"too-many-steps", func(o *QwenImage21Options) { o.Steps = 1001 }},
		{"small-guidance", func(o *QwenImage21Options) { o.GuidanceScale = .5 }},
		{"large-guidance", func(o *QwenImage21Options) { o.GuidanceScale = 21; o.NegativePrompt = "noise" }},
		{"nan-guidance", func(o *QwenImage21Options) { o.GuidanceScale = float32(math.NaN()) }},
		{"infinite-guidance", func(o *QwenImage21Options) { o.GuidanceScale = float32(math.Inf(1)) }},
		{"unused-negative", func(o *QwenImage21Options) { o.NegativePrompt = "noise" }},
		{"missing-negative", func(o *QwenImage21Options) { o.GuidanceScale = 2 }},
		{"image-conditioning", func(o *QwenImage21Options) { o.Prompt = "<|image_pad|>" }},
		{"video-conditioning", func(o *QwenImage21Options) { o.Prompt = "<|video_pad|>" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := qi21SmallOptions()
			tc.change(&opts)
			opts.Progress = func(int, int) { t.Error("invalid request started denoising") }
			if img, err := m.Generate(context.Background(), opts); err == nil || img != nil {
				t.Fatalf("invalid request returned image=%v, error=%v", img != nil, err)
			}
		})
	}
}

func TestQwenImage21ModelCancellationAndReuse(t *testing.T) {
	m := qi21PipelineFixture(t)
	opts := qi21SmallOptions()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if img, err := m.Generate(ctx, opts); !errors.Is(err, context.Canceled) || img != nil {
		t.Fatalf("cancelled request returned image=%v, error=%v", img != nil, err)
	}
	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	steps := 0
	opts.Progress = func(int, int) { steps++; cancel() }
	if img, err := m.Generate(ctx, opts); !errors.Is(err, context.Canceled) || img != nil || steps != 1 {
		t.Fatalf("mid-generation cancellation: image=%v, error=%v, steps=%d", img != nil, err, steps)
	}
	opts.Progress = nil
	if _, err := m.Generate(context.Background(), opts); err != nil {
		t.Fatalf("cancelled request left the model unusable: %v", err)
	}
}

func TestQwenImage21ModelQueuedCancellationAndClose(t *testing.T) {
	m := qi21PipelineFixture(t)
	entered, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	defer unblock()
	opts := qi21SmallOptions()
	opts.Progress = func(completed, total int) {
		if completed == 1 {
			close(entered)
			<-release
		}
	}
	generated := make(chan error, 1)
	go func() { _, err := m.Generate(context.Background(), opts); generated <- err }()
	select {
	case <-entered:
	case <-time.After(10 * time.Second):
		t.Fatal("active generation did not enter its callback")
	}
	// The model gate is held by the first public Generate call. A second
	// request must still observe cancellation while waiting for that gate.
	ctx, cancel := context.WithCancel(context.Background())
	waiting := make(chan error, 1)
	go func() { _, err := m.Generate(ctx, qi21SmallOptions()); waiting <- err }()
	cancel()
	select {
	case err := <-waiting:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("queued cancellation: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("queued request ignored cancellation")
	}
	closed := make(chan error, 1)
	go func() { closed <- m.Close() }()
	select {
	case err := <-closed:
		t.Fatalf("Close returned while generation was active: %v", err)
	default:
	}
	unblock()
	select {
	case err := <-generated:
		if err != nil {
			t.Fatalf("Close interrupted active generation: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("active generation did not complete")
	}
	select {
	case err := <-closed:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Close did not complete")
	}
	if len(m.stores) != 0 || m.encoder != nil || m.transformer != nil || m.vae != nil {
		t.Fatal("closed model retained its components or mappings")
	}
	if img, err := m.Generate(context.Background(), qi21SmallOptions()); err == nil || img != nil {
		t.Fatal("closed model accepted another request")
	}
	if err := m.Close(); err != nil {
		t.Fatalf("repeated Close: %v", err)
	}
}

func TestQwenImage21ModelOpenFailures(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if m, err := OpenQwenImage21(ctx, "does-not-exist"); !errors.Is(err, context.Canceled) || m != nil {
		t.Fatalf("cancelled Open returned model=%v, error=%v", m != nil, err)
	}
	for _, tc := range []struct {
		name string
		edit func(*testing.T, string)
	}{
		{"invalid-index", func(t *testing.T, dir string) {
			if err := os.WriteFile(filepath.Join(dir, "model_index.json"), []byte(`{"_class_name":"QwenImagePipeline"}`), 0600); err != nil {
				t.Fatal(err)
			}
		}},
		{"missing-last-shard", func(t *testing.T, dir string) {
			if err := os.Remove(filepath.Join(dir, "vae/diffusion_pytorch_model.safetensors")); err != nil {
				t.Fatal(err)
			}
		}},
		{"incompatible-components", func(t *testing.T, dir string) {
			for _, name := range []string{"config.json", "diffusion_pytorch_model.safetensors"} {
				data, err := os.ReadFile(filepath.Join("testdata/qwen_image21_transformer", name))
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(dir, "transformer", name), data, 0600); err != nil {
					t.Fatal(err)
				}
			}
		}},
		{"unsupported-scheduler", func(t *testing.T, dir string) {
			path := filepath.Join(dir, "scheduler/scheduler_config.json")
			var cfg qi21SchedulerConfig
			if err := readModelJSON(path, 64<<10, &cfg); err != nil {
				t.Fatal(err)
			}
			cfg.Stochastic = true
			data, err := json.Marshal(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := qi21PipelineFixtureDir(t)
			tc.edit(t, dir)
			if model, err := OpenQwenImage21(context.Background(), dir); err == nil || model != nil {
				if model != nil {
					_ = model.Close()
				}
				t.Fatalf("invalid model returned model=%v, error=%v", model != nil, err)
			}
			// On platforms that prevent deleting mapped files this also catches
			// mappings leaked by a partially completed Open.
			if err := os.RemoveAll(dir); err != nil {
				t.Fatalf("failed Open retained model files: %v", err)
			}
		})
	}
}
