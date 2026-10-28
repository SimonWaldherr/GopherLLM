package engine

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"slices"

	"github.com/SimonWaldherr/GopherLLM/internal/numeric"
)

// DiffusionOptions controls unconditional CIFAR-10 image sampling. The model
// produces 32×32 RGB images; it does not accept text prompts. Steps selects
// deterministic DDIM sampling (eta=0), with 50 steps when zero. Seed fixes the
// initial Gaussian noise. Results can differ slightly between compute backends.
type DiffusionOptions struct {
	Seed  int64
	Steps int
}

// DiffusionModel runs the google/ddpm-cifar10-32 UNet locally. Calls to Generate
// are serialized to bound workspace memory. Close waits for an active call.
// Loaders never download weights or execute model-supplied code.
type DiffusionModel struct {
	gate chan struct{}
	net  *diffusionUNet
}

// OpenDiffusion loads the original F32 safetensors checkpoint and its
// config.json and scheduler_config.json from path. Path may name the directory
// or diffusion_pytorch_model.safetensors. Only the validated CIFAR-10 graph and
// noise schedule are supported; other diffusion architectures are rejected.
func OpenDiffusion(ctx context.Context, path string) (*DiffusionModel, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	st, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("diffusion: %w", err)
	}
	dir := path
	if !st.IsDir() {
		dir = filepath.Dir(path)
	} else {
		path = filepath.Join(path, "diffusion_pytorch_model.safetensors")
	}
	if err := diffusionValidateJSON(filepath.Join(dir, "config.json"), diffusionCIFARConfig, diffusionConfigDefaults); err != nil {
		return nil, err
	}
	if err := diffusionValidateJSON(filepath.Join(dir, "scheduler_config.json"), diffusionCIFARScheduler, diffusionSchedulerDefaults); err != nil {
		return nil, err
	}
	st, err = os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("diffusion: %w", err)
	}
	// The supported checkpoint is 143 MB. Reject directories, devices and
	// implausibly large files before mmap or the portable read-copy fallback.
	if !st.Mode().IsRegular() || st.Size() < 8 || st.Size() > 160<<20 {
		return nil, errors.New("diffusion: expected a regular safetensors file of at most 160 MiB")
	}
	mapping, err := OpenMmap(path)
	if err != nil {
		return nil, fmt.Errorf("diffusion: %w", err)
	}
	defer mapping.Close()
	data := mapping.Bytes()
	// Recheck the opened mapping: the file may have changed after os.Stat.
	if len(data) < 8 || len(data) > 160<<20 || binary.LittleEndian.Uint64(data[:8]) > 1<<20 {
		return nil, errors.New("diffusion: invalid or oversized safetensors header")
	}
	file, err := ParseSafetensors(data)
	if err != nil {
		return nil, err
	}
	loader := diffusionLoader{ctx: ctx, file: file, used: make(map[string]bool)}
	net := loader.unet()
	if loader.err != nil {
		return nil, loader.err
	}
	if len(loader.used) != len(file.Tensors) {
		return nil, errors.New("diffusion: checkpoint contains tensors outside the supported CIFAR-10 graph")
	}
	return &DiffusionModel{gate: make(chan struct{}, 1), net: net}, nil
}

// Close releases the model weights. It is safe to call repeatedly.
func (m *DiffusionModel) Close() error {
	if m == nil || m.gate == nil {
		return nil
	}
	m.gate <- struct{}{}
	m.net = nil
	<-m.gate
	return nil
}

// Generate samples one unconditional 32×32 image. Cancellation is checked
// between network operators and while waiting for another generation.
func (m *DiffusionModel) Generate(ctx context.Context, opts DiffusionOptions) (image.Image, error) {
	if m == nil || m.gate == nil {
		return nil, errors.New("diffusion: model is closed")
	}
	steps := opts.Steps
	if steps == 0 {
		steps = 50
	}
	if steps < 1 || steps > 1000 {
		return nil, errors.New("diffusion: steps must be between 1 and 1000")
	}
	select {
	case m.gate <- struct{}{}:
		defer func() { <-m.gate }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m.net == nil {
		return nil, errors.New("diffusion: model is closed")
	}
	rng := rand.New(rand.NewSource(opts.Seed))
	sample := make([]float32, 3*32*32)
	for i := range sample {
		sample[i] = float32(rng.NormFloat64())
	}
	if err := m.net.sample(ctx, sample, steps); err != nil {
		return nil, err
	}
	out := image.NewNRGBA(image.Rect(0, 0, 32, 32))
	for p := range 32 * 32 {
		for c := range 3 {
			value := sample[c*32*32+p]
			if !finite32(value) {
				return nil, errors.New("diffusion: model produced a non-finite sample")
			}
			out.Pix[p*4+c] = uint8(math.RoundToEven(float64(max(float32(0), min(float32(1), value/2+0.5))) * 255))
		}
		out.Pix[p*4+3] = 255
	}
	return out, nil
}

// Exact graph validation prevents silent acceptance of a checkpoint requiring
// different normalization, conditioning, padding or timestep conventions.
const diffusionCIFARConfig = `{
 "_class_name":"UNet2DModel", "act_fn":"silu", "attention_head_dim":null,
 "block_out_channels":[128,256,256,256], "center_input_sample":false,
 "down_block_types":["DownBlock2D","AttnDownBlock2D","DownBlock2D","DownBlock2D"],
 "downsample_padding":0, "flip_sin_to_cos":false, "freq_shift":1,
 "in_channels":3, "layers_per_block":2, "mid_block_scale_factor":1,
 "norm_eps":0.000001, "norm_num_groups":32, "out_channels":3,
 "sample_size":32, "time_embedding_type":"positional",
 "up_block_types":["UpBlock2D","UpBlock2D","AttnUpBlock2D","UpBlock2D"]
}`

const diffusionConfigDefaults = `{
 "add_attention":true, "attn_norm_num_groups":null, "class_embed_type":null,
 "num_class_embeds":null, "num_train_timesteps":null, "dropout":0,
 "downsample_type":"conv", "upsample_type":"conv", "mid_block_type":"UNetMidBlock2D",
 "resnet_time_scale_shift":"default", "time_embedding_dim":null
}`

const diffusionCIFARScheduler = `{
 "_class_name":"DDPMScheduler", "beta_end":0.02, "beta_schedule":"linear",
 "beta_start":0.0001, "clip_sample":true, "num_train_timesteps":1000,
 "trained_betas":null, "variance_type":"fixed_large"
}`

const diffusionSchedulerDefaults = `{
 "prediction_type":"epsilon", "thresholding":false, "dynamic_thresholding_ratio":0.995,
 "clip_sample_range":1, "sample_max_value":1, "timestep_spacing":"leading",
 "steps_offset":0, "rescale_betas_zero_snr":false
}`

func diffusionValidateJSON(path, requiredJSON, defaultsJSON string) error {
	var got, required, defaults map[string]any
	if err := readModelJSON(path, 64<<10, &got); err != nil {
		return fmt.Errorf("diffusion: %w", err)
	}
	_ = json.Unmarshal([]byte(requiredJSON), &required)
	_ = json.Unmarshal([]byte(defaultsJSON), &defaults)
	for key, expected := range required {
		if value, ok := got[key]; !ok || !reflect.DeepEqual(value, expected) {
			return fmt.Errorf("diffusion: unsupported %s value for %s", filepath.Base(path), key)
		}
	}
	for key, value := range got {
		if _, ok := required[key]; ok {
			continue
		}
		if key == "_diffusers_version" || key == "_name_or_path" || key == "_use_default_values" {
			continue
		}
		if expected, ok := defaults[key]; !ok || !reflect.DeepEqual(value, expected) {
			return fmt.Errorf("diffusion: unsupported %s field %s", filepath.Base(path), key)
		}
	}
	return nil
}

type diffusionLoader struct {
	ctx  context.Context
	file *SafetensorsFile
	used map[string]bool
	err  error
}

func (l *diffusionLoader) tensor(name string, shape ...int) []float32 {
	if l.err != nil {
		return nil
	}
	if l.err = l.ctx.Err(); l.err != nil {
		return nil
	}
	info, ok := l.file.Tensors[name]
	if !ok || info.DType != SafetensorsF32 || !slices.Equal(info.Shape, shape) {
		l.err = fmt.Errorf("diffusion: tensor %s must be F32 with shape %v", name, shape)
		return nil
	}
	// Shapes are checked against fixed dimensions before the parser allocates
	// or converts data, including any overflowed Numel in malformed headers.
	numel := 1
	for _, dim := range shape {
		numel *= dim
	}
	if info.Numel != numel || info.Begin < 0 || info.End < info.Begin || info.End-info.Begin != numel*4 || info.End > len(l.file.Data)-l.file.DataOffset {
		l.err = fmt.Errorf("diffusion: invalid byte range for tensor %s", name)
		return nil
	}
	values, err := l.file.F32(name)
	if err != nil {
		l.err = err
		return nil
	}
	if !numeric.AllFiniteF32(values) {
		l.err = fmt.Errorf("diffusion: non-finite weight in %s", name)
		return nil
	}
	l.used[name] = true
	return values
}

func (l *diffusionLoader) conv(name string, in, out, kernel int) diffusionConv {
	return diffusionConv{in: in, out: out, kernel: kernel,
		weight: l.tensor(name+".weight", out, in, kernel, kernel), bias: l.tensor(name+".bias", out)}
}

func (l *diffusionLoader) linear(name string, in, out int) diffusionConv {
	return diffusionConv{in: in, out: out, kernel: 1,
		weight: l.tensor(name+".weight", out, in), bias: l.tensor(name+".bias", out)}
}

func (l *diffusionLoader) norm(name string, channels int) diffusionNorm {
	return diffusionNorm{weight: l.tensor(name+".weight", channels), bias: l.tensor(name+".bias", channels)}
}

func (l *diffusionLoader) resnet(name string, in, out int) diffusionResnet {
	r := diffusionResnet{norm1: l.norm(name+".norm1", in), norm2: l.norm(name+".norm2", out),
		conv1: l.conv(name+".conv1", in, out, 3), conv2: l.conv(name+".conv2", out, out, 3),
		time: l.linear(name+".time_emb_proj", 512, out)}
	if in != out {
		shortcut := l.conv(name+".conv_shortcut", in, out, 1)
		r.shortcut = &shortcut
	}
	return r
}

func (l *diffusionLoader) attention(name string, channels int) *diffusionAttention {
	return &diffusionAttention{norm: l.norm(name+".group_norm", channels),
		query: l.linear(name+".query", channels, channels), key: l.linear(name+".key", channels, channels),
		value: l.linear(name+".value", channels, channels), out: l.linear(name+".proj_attn", channels, channels)}
}

func (l *diffusionLoader) unet() *diffusionUNet {
	n := &diffusionUNet{input: l.conv("conv_in", 3, 128, 3), output: l.conv("conv_out", 128, 3, 3),
		outputNorm: l.norm("conv_norm_out", 128), time1: l.linear("time_embedding.linear_1", 128, 512),
		time2: l.linear("time_embedding.linear_2", 512, 512)}
	channels := []int{128, 256, 256, 256}
	previous := 128
	skips := []int{128}
	for stage, out := range channels {
		block := &n.down[stage]
		for layer := range 2 {
			name := fmt.Sprintf("down_blocks.%d", stage)
			block.resnets = append(block.resnets, l.resnet(fmt.Sprintf("%s.resnets.%d", name, layer), previous, out))
			if stage == 1 {
				block.attentions = append(block.attentions, l.attention(fmt.Sprintf("%s.attentions.%d", name, layer), out))
			}
			previous = out
			skips = append(skips, out)
		}
		if stage < 3 {
			conv := l.conv(fmt.Sprintf("down_blocks.%d.downsamplers.0.conv", stage), out, out, 3)
			block.resample = &conv
			skips = append(skips, out)
		}
	}
	n.mid1 = l.resnet("mid_block.resnets.0", 256, 256)
	n.midAttention = l.attention("mid_block.attentions.0", 256)
	n.mid2 = l.resnet("mid_block.resnets.1", 256, 256)
	for stage := range 4 {
		out := channels[3-stage]
		block := &n.up[stage]
		for layer := range 3 {
			in := previous + skips[len(skips)-1]
			skips = skips[:len(skips)-1]
			name := fmt.Sprintf("up_blocks.%d", stage)
			block.resnets = append(block.resnets, l.resnet(fmt.Sprintf("%s.resnets.%d", name, layer), in, out))
			if stage == 2 {
				block.attentions = append(block.attentions, l.attention(fmt.Sprintf("%s.attentions.%d", name, layer), out))
			}
			previous = out
		}
		if stage < 3 {
			conv := l.conv(fmt.Sprintf("up_blocks.%d.upsamplers.0.conv", stage), out, out, 3)
			block.resample = &conv
		}
	}
	return n
}
