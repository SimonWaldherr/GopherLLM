package gopherllm

import (
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"image/color"
	"image/png"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

type flux3Reference struct {
	Encoder []struct {
		Prompt, Key string
		Tokens      []uint32
		Real        int
	}
	Transformer []struct {
		H, W         int
		Flow         bool
		Time, Target float32
		Key          string
	}
	Pipeline []struct {
		H, W        int
		Flow        bool
		Prompt, Key string
	}
}

func flux3Fixture(t testing.TB) (*Flux3ImageKleinModel, *qi21Weights, flux3Reference, string) {
	t.Helper()
	dir := t.TempDir()
	root := filepath.Join("testdata", "flux3")
	for _, name := range []string{"model.safetensors", "text_encoder/config.json", "text_encoder/tokenizer.json", "text_encoder/model.safetensors"} {
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(dir, name)
		if err = os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	f, err := os.Open(filepath.Join(root, "ae.safetensors.gz"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	defer z.Close()
	data, err := io.ReadAll(io.LimitReader(z, 32<<20))
	if err != nil || len(data) == 32<<20 {
		t.Fatalf("VAE fixture: %v", err)
	}
	if err = os.WriteFile(filepath.Join(dir, "ae.safetensors"), data, 0600); err != nil {
		t.Fatal(err)
	}
	m, err := OpenImageModel(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	model, ok := m.(*Flux3ImageKleinModel)
	if !ok {
		t.Fatalf("dispatched %T", m)
	}
	t.Cleanup(func() { _ = model.Close() })
	source, err := qi21OpenWeights(context.Background(), root, "reference.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(source.close)
	var ref flux3Reference
	if err = readModelJSON(filepath.Join(root, "reference.json"), 1<<20, &ref); err != nil {
		t.Fatal(err)
	}
	return model, source, ref, dir
}

func TestFlux3TextRoPE(t *testing.T) {
	for _, dim := range []int{8, 128} {
		for _, real := range []int{1, 80, 4095} {
			const theta = 5000000
			cs, sn := flux3TextRoPE(real+1, real, dim, theta)
			for _, token := range []int{0, real - 1, real} {
				position := token
				if token == real {
					position = 1
				}
				x := benchFloatSlice(3 * dim)
				want := slices.Clone(x)
				for j := range dim / 2 {
					s, c := math.Sincos(float64(position) / math.Pow(theta, float64(2*j)/float64(dim)))
					for head := range 3 {
						p, q := head*dim+j, head*dim+j+dim/2
						a, b := want[p], want[q]
						want[p], want[q] = a*float32(c)-b*float32(s), a*float32(s)+b*float32(c)
					}
				}
				applyPreparedRope(x, dim, 3, dim/2, dim/2, sn[token*dim/2:], cs[token*dim/2:], false)
				qi21AssertNear(t, x, want, 5e-7)
			}
		}
	}
}

func TestFlux3CompactSwiGLU(t *testing.T) {
	for _, rows := range []int{1, 2, 3, qi21TokenTile} {
		for _, hidden := range []int{8, 33, 129} {
			storage := benchFloatSlice(rows*hidden*2 + 2)
			left, right := storage[0], storage[len(storage)-1]
			input := storage[1 : len(storage)-1]
			want := make([]float32, rows*hidden)
			for i := range rows {
				row := input[i*2*hidden : (i+1)*2*hidden]
				siluMulF32(row[:hidden], row[hidden:], want[i*hidden:(i+1)*hidden])
			}
			got := flux3CompactSwiGLU(input, hidden)
			if !slices.Equal(got, want) || storage[0] != left || storage[len(storage)-1] != right {
				t.Fatalf("SwiGLU compaction corrupted data: rows=%d, hidden=%d", rows, hidden)
			}
		}
	}
}

func TestFlux3Reference(t *testing.T) {
	m, source, ref, _ := flux3Fixture(t)
	read := func(name string, shape ...int) []float32 {
		t.Helper()
		v, e := source.floats(name, shape...)
		if e != nil {
			t.Fatal(e)
		}
		return v
	}
	for _, tc := range ref.Encoder {
		t.Run(tc.Key, func(t *testing.T) {
			got, err := m.encoder.encode(context.Background(), tc.Prompt)
			if err != nil {
				t.Fatal(err)
			}
			want := read(tc.Key, len(tc.Tokens), 128)
			qi21AssertNear(t, got, want, 3e-5)
			if len(tc.Tokens) > tc.Real+1 {
				for i := tc.Real + 1; i < len(tc.Tokens); i++ {
					if !slices.Equal(got[tc.Real*128:(tc.Real+1)*128], got[i*128:(i+1)*128]) {
						t.Fatal("padding features differ")
					}
				}
			}
		})
	}
	for _, tc := range ref.Transformer {
		t.Run(tc.Key, func(t *testing.T) {
			m.transformer.flowMap = tc.Flow
			s, err := m.transformer.newState(read(tc.Key+".text", 9, 128), tc.H, tc.W)
			if err != nil {
				t.Fatal(err)
			}
			noise := read(tc.Key+".noise", tc.H*tc.W, 128)
			got, err := s.predict(context.Background(), noise, tc.Time, tc.Target)
			if err != nil {
				t.Fatal(err)
			}
			qi21AssertNear(t, got, read(tc.Key+".expected", tc.H*tc.W, 128), 3e-5)
			copyGot := slices.Clone(got)
			again, err := s.predict(context.Background(), noise, tc.Time, tc.Target)
			if err != nil {
				t.Fatal(err)
			}
			qi21AssertNear(t, again, copyGot, 1e-7)
		})
	}
	for _, steps := range []int{1, 3, 16, 50} {
		got, err := flux3Schedule(steps)
		if err != nil {
			t.Fatal(err)
		}
		qi21AssertNear(t, got, read(fmt.Sprintf("schedule.%d", steps), steps+1), 2e-7)
	}
	for _, tc := range ref.Pipeline {
		t.Run(tc.Key, func(t *testing.T) {
			m.transformer.flowMap = tc.Flow
			noise := read(tc.Key+".noise", tc.H*tc.W, 128)
			original := slices.Clone(noise)
			progress := 0
			got, err := m.generate(context.Background(), ImageGenerationOptions{Prompt: tc.Prompt, Width: tc.W * 16, Height: tc.H * 16, Steps: 2, Progress: func(done, total int) {
				progress++
				if done != progress || total != 2 {
					t.Errorf("progress %d/%d", done, total)
				}
			}}, noise)
			if err != nil {
				t.Fatal(err)
			}
			if progress != 2 || !slices.Equal(noise, original) {
				t.Fatal("inconsistent steps or caller-owned noise modified")
			}
			f, err := os.Open(filepath.Join("testdata", "flux3", tc.Key+".png"))
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			want, err := png.Decode(f)
			if err != nil {
				t.Fatal(err)
			}
			if got.Bounds() != want.Bounds() {
				t.Fatal("wrong image dimensions")
			}
			for y := range got.Bounds().Dy() {
				for x := range got.Bounds().Dx() {
					a, b := got.NRGBAAt(x, y), color.NRGBAModel.Convert(want.At(x, y)).(color.NRGBA)
					for _, pair := range [][2]uint8{{a.R, b.R}, {a.G, b.G}, {a.B, b.B}, {a.A, b.A}} {
						if math.Abs(float64(pair[0])-float64(pair[1])) > 1 {
							t.Fatalf("pixel (%d,%d) got %v, want %v", x, y, a, b)
						}
					}
				}
			}
		})
	}
}

func TestFlux3LifecycleAndValidation(t *testing.T) {
	m, _, _, dir := flux3Fixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.Generate(ctx, ImageGenerationOptions{Prompt: "cube"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	for _, opts := range []ImageGenerationOptions{{Prompt: " "}, {Prompt: "cube", Width: 65}, {Prompt: "cube", Steps: -1}, {Prompt: "cube", GuidanceScale: 4}, {Prompt: "cube", NegativePrompt: "bad"}} {
		if _, err := m.Generate(context.Background(), opts); err == nil {
			t.Fatalf("accepted %+v", opts)
		}
	}
	large := *m.transformer
	large.dim = 8192
	if _, err := large.newState(make([]float32, 4096*128), 128, 128); err == nil {
		t.Fatal("accepted oversized state")
	}
	if _, err := m.vae.decode(context.Background(), []float32{1}, 4, 4); err == nil {
		t.Fatal("accepted malformed latents")
	}
	if _, err := flux3Schedule(1001); err == nil {
		t.Fatal("accepted too many steps")
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Generate(context.Background(), ImageGenerationOptions{Prompt: "cube"}); err == nil {
		t.Fatal("generation after close")
	}
	if len(m.stores) != 0 || m.encoder != nil || m.vae != nil {
		t.Fatal("closed model retains components")
	}
	if err := os.WriteFile(filepath.Join(dir, "model.gguf"), []byte("duplicate"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenImageModel(context.Background(), dir); err == nil {
		t.Fatal("accepted ambiguous checkpoint")
	}
}

func TestOpenImageModelQwenDispatch(t *testing.T) {
	m, err := OpenImageModel(context.Background(), qi21PipelineFixtureDir(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if _, ok := m.(*QwenImage21Model); !ok {
		t.Fatalf("dispatched %T", m)
	}
	m, err = OpenImageModel(context.Background(), t.TempDir())
	if err == nil || m != nil {
		t.Fatalf("failed open returned model: %T %v", m, err)
	}
}

// Match the range importer's compact layout, including the omitted encoder
// tail. Both base and flow-map checkpoints must be inferred from their weights.
func TestFlux3CompactWeightsAndCancellation(t *testing.T) {
	m, _, _, dir := flux3Fixture(t)
	convert := func(source *qi21Weights, target string, text, flow bool) {
		t.Helper()
		var tensors []ggufTensor
		for name, tensor := range source.tensors {
			if text && name == "model.language_model.norm.weight" {
				continue
			}
			if text && strings.HasPrefix(name, "model.language_model.layers.") {
				i, err := strconv.Atoi(strings.Split(name, ".")[3])
				if err != nil {
					t.Fatal(err)
				}
				if i >= 32 {
					continue
				}
			}
			if !flow && strings.HasPrefix(name, "ref_time_in.") {
				continue
			}
			dims := make([]uint64, len(tensor.shape))
			for i, v := range tensor.shape {
				dims[len(dims)-i-1] = uint64(v)
			}
			typ, raw := tensor.w.Type, tensor.w.Raw
			if len(tensor.shape) == 2 && tensor.w.Cols%32 == 0 {
				f, ok := tensorToF32(typ, raw, tensor.w.Rows*tensor.w.Cols)
				if !ok {
					t.Fatal("decode fixture")
				}
				raw = QuantizeRowQ8_0(f, len(f))
				typ = GGMLTypeQ8_0
			}
			tensors = append(tensors, ggufTensor{name, dims, typ, raw})
		}
		if err := os.WriteFile(target, buildGGUF(3, nil, tensors), 0600); err != nil {
			t.Fatal(err)
		}
	}
	convert(m.stores[0], filepath.Join(dir, "text_encoder", "model.gguf"), true, true)
	convert(m.stores[1], filepath.Join(dir, "model.gguf"), false, true)
	convert(m.stores[1], filepath.Join(dir, "model-base.gguf"), false, false)
	// Windows cannot remove a mapped source file. Finish conversion and close
	// all source mappings before replacing the fixture's checkpoint files.
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(dir, "text_encoder", "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	for _, flow := range []bool{true, false} {
		t.Run(fmt.Sprintf("flow=%t", flow), func(t *testing.T) {
			if !flow {
				if err := os.Remove(filepath.Join(dir, "model.gguf")); err != nil {
					t.Fatal(err)
				}
				if err := os.Rename(filepath.Join(dir, "model-base.gguf"), filepath.Join(dir, "model.gguf")); err != nil {
					t.Fatal(err)
				}
			}
			loaded, err := OpenFlux3ImageKlein(context.Background(), dir)
			if err != nil {
				t.Fatal(err)
			}
			defer loaded.Close()
			if loaded.transformer.flowMap != flow {
				t.Fatal("wrong checkpoint variant")
			}
			if loaded.encoder.core.config.NLayers != 32 {
				t.Fatal("unused text layers loaded")
			}
			if loaded.transformer.image.Type != GGMLTypeQ8_0 {
				t.Fatal("matrix lost packed storage")
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			calls := 0
			_, err = loaded.Generate(ctx, ImageGenerationOptions{Prompt: "cube", Width: 64, Height: 64, Progress: func(done, total int) {
				calls++
				want := 16
				if !flow {
					want = 50
				}
				if done != 1 || total != want {
					t.Errorf("schedule %d/%d", done, total)
				}
				cancel()
			}})
			if !errors.Is(err, context.Canceled) || calls != 1 {
				t.Fatalf("mid-generation cancellation: %v, calls=%d", err, calls)
			}
			img, err := loaded.Generate(context.Background(), ImageGenerationOptions{Prompt: "cube", Width: 64, Height: 64, Steps: 1})
			if err != nil {
				t.Fatal(err)
			}
			if img.Bounds().Dx() != 64 || img.NRGBAAt(0, 0).A != 255 {
				t.Fatal("invalid RGB image")
			}
		})
	}
}
