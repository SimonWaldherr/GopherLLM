package engine

import (
	"context"
	"encoding/json"
	"fmt"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func qi21PipelineFixture(t *testing.T) *QwenImage21Model {
	t.Helper()
	model, err := OpenQwenImage21(context.Background(), qi21PipelineFixtureDir(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = model.Close() })
	return model
}

func qi21PipelineFixtureDir(t *testing.T) string {
	t.Helper()
	root := filepath.Join("../../testdata", "qwen_image21_pipeline")
	dir := t.TempDir()
	write := func(name string, data []byte) {
		t.Helper()
		path := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	copyFile := func(target, source string) {
		t.Helper()
		data, err := os.ReadFile(source)
		if err != nil {
			t.Fatal(err)
		}
		write(target, data)
	}
	for _, name := range []string{"model_index.json", "scheduler/scheduler_config.json", "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors"} {
		copyFile(name, filepath.Join(root, name))
	}
	var encoder struct{ Config, Tokenizer json.RawMessage }
	if err := readModelJSON("../../testdata/qwen_image21_encoder.json", 128<<10, &encoder); err != nil {
		t.Fatal(err)
	}
	write("text_encoder/config.json", encoder.Config)
	write("processor/tokenizer.json", encoder.Tokenizer)
	copyFile("text_encoder/model.safetensors", "../../testdata/qwen_image21_encoder.safetensors")
	copyFile("vae/config.json", "../../testdata/qwen_image21_vae_config.json")
	copyFile("vae/diffusion_pytorch_model.safetensors", "../../testdata/qwen_image21_vae.safetensors")
	return dir
}

// This exercises the public model loader and the complete native tokenizer,
// text encoder, eight cached DiT/Euler steps, latent layout and RGBA decoder.
// The Python oracle composes the independently verified official components.
func TestQwenImage21PipelineReference(t *testing.T) {
	model := qi21PipelineFixture(t)
	root := filepath.Join("../../testdata", "qwen_image21_pipeline")
	var reference struct {
		Prompt string
		Cases  []struct {
			Width, Height int
			Image         string
			Noise         []float32
		}
	}
	if err := readModelJSON(filepath.Join(root, "reference.json"), 1<<20, &reference); err != nil {
		t.Fatal(err)
	}
	for _, tc := range reference.Cases {
		t.Run(fmt.Sprintf("%dx%d", tc.Width, tc.Height), func(t *testing.T) {
			progress := 0
			noise := slices.Clone(tc.Noise)
			got, err := model.generate(context.Background(), QwenImage21Options{
				Prompt: reference.Prompt, Width: tc.Width, Height: tc.Height,
				Progress: func(completed, total int) {
					progress++
					if completed != progress || total != 8 {
						t.Errorf("progress %d/%d, expected %d/8", completed, total, progress)
					}
				},
			}, noise)
			if err != nil {
				t.Fatal(err)
			}
			if progress != 8 || !slices.Equal(noise, tc.Noise) {
				t.Fatal("generation lost steps or modified caller-owned noise")
			}
			f, err := os.Open(filepath.Join(root, tc.Image))
			if err != nil {
				t.Fatal(err)
			}
			want, err := png.Decode(f)
			_ = f.Close()
			if err != nil {
				t.Fatal(err)
			}
			if got.Bounds() != want.Bounds() {
				t.Fatalf("image dimensions %v, expected %v", got.Bounds(), want.Bounds())
			}
			var maximum, different int
			for y := 0; y < tc.Height; y++ {
				for x := 0; x < tc.Width; x++ {
					a, b := got.NRGBAAt(x, y), color.NRGBAModel.Convert(want.At(x, y)).(color.NRGBA)
					for c, pair := range [][2]uint8{{a.R, b.R}, {a.G, b.G}, {a.B, b.B}, {a.A, b.A}} {
						delta := max(int(pair[0])-int(pair[1]), int(pair[1])-int(pair[0]))
						maximum = max(maximum, delta)
						if delta != 0 {
							different++
						}
						if delta > 1 {
							t.Fatalf("pixel (%d,%d), channel %d: %d, expected %d", x, y, c, pair[0], pair[1])
						}
					}
				}
			}
			t.Logf("eight-step RGBA oracle: maximum byte error %d, %d differing channels", maximum, different)
		})
	}
}
