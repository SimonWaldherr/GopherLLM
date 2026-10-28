package engine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image/color"
	"image/png"
	"math"
	"os"
	"slices"
	"testing"
)

type qi21VAEFixtureSource struct{ file *SafetensorsFile }

func (s qi21VAEFixtureSource) matrix(name string, rows, cols int) (Weight, error) {
	v, err := s.floats(name, rows, cols)
	return Weight{F32: v, Rows: rows, Cols: cols, Type: GGMLTypeF32}, err
}
func (s qi21VAEFixtureSource) floats(name string, shape ...int) ([]float32, error) {
	info, ok := s.file.Tensors[name]
	if !ok || !slices.Equal(info.Shape, shape) {
		return nil, fmt.Errorf("fixture tensor %s shape %v, want %v", name, info.Shape, shape)
	}
	typ := GGMLTypeF32
	if info.DType == SafetensorsBF16 {
		typ = GGMLTypeBF16
	}
	v, valid := tensorToF32(typ, s.file.Data[s.file.DataOffset+info.Begin:s.file.DataOffset+info.End], info.Numel)
	if !valid {
		return nil, fmt.Errorf("cannot decode fixture tensor %s", name)
	}
	return v, nil
}
func qi21VAEFixture(t *testing.T) (*qwenImage21VAE, qi21VAEFixtureSource) {
	t.Helper()
	data, err := os.ReadFile("../../testdata/qwen_image21_vae.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	file, err := ParseSafetensors(data)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("../../testdata/qwen_image21_vae_config.json")
	if err != nil {
		t.Fatal(err)
	}
	var cfg qwenImage21VAEConfig
	if err = json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	src := qi21VAEFixtureSource{file}
	v, err := loadQwenImage21VAE(context.Background(), src, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return v, src
}

func TestQwenImage21VAEDecoderReference(t *testing.T) {
	v, src := qi21VAEFixture(t)
	for _, tc := range []struct {
		name string
		h, w int
	}{{"small", 2, 3}, {"tiled", 17, 13}} {
		t.Run(tc.name, func(t *testing.T) {
			latents, err := src.floats(tc.name+".latents", tc.h*tc.w, 64)
			if err != nil {
				t.Fatal(err)
			}
			got, err := v.decode(context.Background(), latents, tc.h, tc.w)
			if err != nil {
				t.Fatal(err)
			}
			f, err := os.Open("../../testdata/qwen_image21_vae_" + tc.name + ".png")
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			want, err := png.Decode(f)
			if err != nil {
				t.Fatal(err)
			}
			if got.Bounds() != want.Bounds() {
				t.Fatalf("bounds %v, want %v", got.Bounds(), want.Bounds())
			}
			maxError := 0
			for y := range got.Bounds().Dy() {
				for x := range got.Bounds().Dx() {
					// NRGBA values preserve straight RGB when alpha is not opaque.
					w := color.NRGBAModel.Convert(want.At(x, y)).(color.NRGBA)
					g := got.NRGBAAt(x, y)
					for c, actual := range []uint8{g.R, g.G, g.B, g.A} {
						expected := []uint8{w.R, w.G, w.B, w.A}[c]
						maxError = max(maxError, int(math.Abs(float64(actual)-float64(expected))))
					}
				}
			}
			if maxError > 1 {
				t.Fatalf("maximum RGBA error %d levels, want <=1", maxError)
			}
			if tc.name == "small" {
				input := diffusionTensor{make([]float32, len(latents)), 64, tc.h, tc.w}
				for ch := range 64 {
					for p := range tc.h * tc.w {
						input.data[ch*tc.h*tc.w+p] = latents[p*64+ch]*v.config.LatentsStd[ch] + v.config.LatentsMean[ch]
					}
				}
				s := &qi21VAEWorkspace{ctx: context.Background()}
				actual := v.decodeTile(s, input)
				if s.err != nil {
					t.Fatal(s.err)
				}
				expected, err := src.floats("small.output", 4, tc.h*16, tc.w*16)
				if err != nil {
					t.Fatal(err)
				}
				maxDelta := float64(0)
				for i, a := range actual.data {
					maxDelta = max(maxDelta, math.Abs(float64(max(float32(-1), min(float32(1), a))-expected[i])))
				}
				if maxDelta > 1e-4 {
					t.Fatalf("float decoder max error %.8g, want <=1e-4", maxDelta)
				}
			}
		})
	}
}

func TestQwenImage21VAERejectsInvalidInputs(t *testing.T) {
	v, src := qi21VAEFixture(t)
	cfg := v.config
	cfg.IsResidual = false
	if _, err := loadQwenImage21VAE(context.Background(), src, cfg); err == nil {
		t.Fatal("accepted unsupported decoder topology")
	}
	cfg = v.config
	cfg.LatentsStd = slices.Clone(cfg.LatentsStd)
	cfg.LatentsStd[0] = 0
	if _, err := loadQwenImage21VAE(context.Background(), src, cfg); err == nil {
		t.Fatal("accepted invalid latent scaling")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := v.decode(ctx, nil, 1, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	if _, err := v.decode(context.Background(), make([]float32, 63), 1, 1); err == nil {
		t.Fatal("accepted wrong latent length")
	}
	latent := make([]float32, 64)
	latent[0] = float32(math.NaN())
	if _, err := v.decode(context.Background(), latent, 1, 1); err == nil {
		t.Fatal("accepted non-finite latents")
	}
	s := &qi21VAEWorkspace{ctx: ctx}
	v.decodeTile(s, diffusionTensor{make([]float32, 64), 64, 1, 1})
	if !errors.Is(s.err, context.Canceled) {
		t.Fatalf("operator cancellation: %v", s.err)
	}
}

type qi21VAECancelSource struct {
	qi21WeightSource
	cancel          context.CancelFunc
	calls, cancelAt int
}

func (s *qi21VAECancelSource) floats(name string, shape ...int) ([]float32, error) {
	s.calls++
	values, err := s.qi21WeightSource.floats(name, shape...)
	if s.calls == s.cancelAt {
		s.cancel()
	}
	return values, err
}

func TestQwenImage21VAELoadCancellation(t *testing.T) {
	v, src := qi21VAEFixture(t)
	for _, cancelAt := range []int{0, 1, 10} {
		t.Run(fmt.Sprintf("after-%d-tensors", cancelAt), func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if cancelAt == 0 {
				cancel()
			}
			controlled := &qi21VAECancelSource{qi21WeightSource: src, cancel: cancel, cancelAt: cancelAt}
			loaded, err := loadQwenImage21VAE(ctx, controlled, v.config)
			if !errors.Is(err, context.Canceled) || loaded != nil {
				t.Fatalf("cancelled load returned decoder=%v, error=%v", loaded != nil, err)
			}
			if controlled.calls != cancelAt {
				t.Fatalf("expanded %d tensors, cancellation occurred after %d", controlled.calls, cancelAt)
			}
		})
	}
}
