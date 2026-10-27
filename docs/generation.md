# Native speech and image generation

Generation uses local Go APIs and standalone commands:

| Capability | Supported checkpoint | Output |
| --- | --- | --- |
| German text-to-speech | `facebook/mms-tts-deu` | Mono 16 kHz PCM/WAV |
| Unconditional image generation | `google/ddpm-cifar10-32` | 32×32 RGB/PNG |
| Text-to-image generation | Qwen-Image-2.1 / Turbo | RGBA/PNG |
| Text-to-image generation (experimental) | FLUX 3 Image Klein / base | RGB/PNG |

Inference runs in the GopherLLM process. Neither command starts Python or
downloads weights. Matrix operations use the existing Accelerate backend on
macOS with CGO and Go/SIMD kernels otherwise. These paths do not yet use Metal
or Core ML. The Go module still has no third-party dependencies.

Qwen-Image-2.1 and FLUX 3 Image Klein have native text-to-image paths described
below. Image editing, reference-image conditioning and video generation are not implemented.
The CIFAR model remains a small unconditional diffusion reference. These APIs
are not yet exposed through the mobile package or language bindings. Text-to-image
generation is available through the local server and its Web UI.

## Qwen-Image-2.1

The native pipeline implements the Qwen3-VL text encoder, 32-layer single-stream
DiT, Flow Matching Euler sampler and RGBA VAE. It preserves the unnormalized
text-encoder features, mixed causal/bidirectional attention and compact prefix
KV reuse across denoising steps. Turbo uses its saved eight-sigma schedule;
passing a different step count is rejected instead of silently changing it.

Use Python 3.11 or newer with NumPy to download the selected official revision
and convert large matrices to Q8_0:

```sh
python3 -m venv .cache/qwen-download
.cache/qwen-download/bin/pip install numpy
.cache/qwen-download/bin/python scripts/download_qwen_image21.py \
  .cache/qwen-image21/turbo --variant turbo
```

The importer reads HTTP ranges and writes GGUF component shards directly. It
does not keep a second full copy of the original checkpoint. Norms and VAE
convolutions retain their original precision; unused vision and LM-head weights
are omitted. The Turbo component files occupy about 15.16 GiB. Reserve extra
free space for normal system operation, particularly swap on a busy Mac.
`--workers 16` raises the bounded download concurrency from its default of four.
`--variant base` selects the original 40-step model. The downloader
pins the upstream revision and retains the model's LICENSE and provenance.
Weight terms are separate from this repository's code license.

```sh
CGO_ENABLED=1 go run ./cmd/image-generate \
  -model .cache/qwen-image21/turbo \
  -prompt 'A red ceramic teapot on a pale blue table, studio photograph.' \
  -width 256 -height 256 -seed 42 -threads 8 -out teapot.png
```

The Go runtime performs inference itself; Python and NumPy are only used for
offline download/conversion and reference fixtures. On macOS, CGO enables the
existing Accelerate matrix kernels. This image pipeline does not yet offload
the DiT or VAE to Metal/Core ML. Large matrices remain mapped, FFNs use bounded
token slabs, and VAE decoding uses overlapping spatial tiles above 256 pixels.
Requests whose estimated DiT state exceeds 2 GiB are rejected before allocating
it; very long prompts may therefore need shortening.

```go
model, err := gopherllm.OpenQwenImage21(ctx, "/path/to/qwen-image21")
if err != nil { return err }
defer model.Close()
img, err := model.Generate(ctx, gopherllm.QwenImage21Options{
    Prompt: "A red ceramic teapot on a pale blue table.",
    Width: 256, Height: 256, Seed: 42,
})
if err != nil { return err }
return png.Encode(writer, img)
```

Original Diffusers safetensors directories are also accepted. The default is
256×256 to keep initial local runs manageable, rather than the official 2K
default. Dimensions must be multiples of 32, from 32 through 2048. The returned
image has straight RGBA channels. For transparency, use the official prompt
form: `This is an RGBA image with transparency. ... The image has alpha channel
and the background is transparent.` A seed reproduces Go's noise sequence, not
PyTorch's. Q8_0 is approximate; support for the graph does not imply bitwise
equivalence to a BF16 reference or fast 2K generation on a 32 GB Mac.

Text-to-image is the implemented scope. Image editing, multiple references,
prompt-rewriting models, LoRA adapters and quantized formats other than those
handled by this importer have not been integrated. See the
[validation report](../benchmarks/generative/qwen-image21.md) for tested paths.

## FLUX 3 Image Klein

The native Go implementation supports text-to-image generation with the BFL
base and distilled checkpoints. It stacks eight intermediate Qwen3-VL hidden
states and includes padding masks, separate image/text streams, joint attention, four-axis
RoPE, flow-map target-time conditioning, Euler sampling and FLUX RGB decoder.
The pipeline is **experimental**: small official-reference tests pass, but a
pretrained image test is pending access to the gated weights. See the
[validation record](../benchmarks/generative/flux3.md).

First obtain access to [the official checkpoint](https://huggingface.co/black-forest-labs/flux-3-image-klein)
and accept its terms in your own Hugging Face account. Then authenticate locally
and download the pinned distilled revision:

```sh
python3 -m venv .cache/flux3-download
.cache/flux3-download/bin/pip install numpy huggingface_hub
.cache/flux3-download/bin/hf auth login
.cache/flux3-download/bin/python scripts/download_flux3.py .cache/flux3/klein

CGO_ENABLED=1 go run ./cmd/image-generate \
  -model .cache/flux3/klein -width 256 -height 256 -seed 42 -threads 8 \
  -prompt 'A studio photograph of a red ceramic teapot on a pale blue table. The teapot is centered in the frame, with a curved handle on the left and a short spout pointing right. Soft light from a large window reveals the glossy glaze. The background is plain cream, with a gentle shadow beneath the teapot.' \
  -out flux3-teapot.png
```

The importer shares Qwen's bounded range reader and Q8_0 converter. It keeps
only text-encoder layers 1–32, omits the unused last four layers, final norm,
vision/LM-head/reference-image branches and loads only the VAE decoder.
Credentials are sent only to Hugging Face, not to
redirected CDN hosts. The downloaded `LICENSE.md` and `source.json` preserve
the checkpoint terms and revision. No Python or network service is used during
inference.

The loader also accepts original BFL `model.safetensors`, `ae.safetensors` and
`text_encoder/` files. Diffusers-renamed weights, FP8/NVFP4 CUDA variants, image
editing, layout conditioning and LoRA are not supported. Supply a detailed scene
caption; the separate 9B prompt upsampler is not loaded automatically.

Defaults are 256×256 and 16 steps for the distilled model; the base model uses
50 steps and guidance 4. `-guidance 1` disables CFG for the base model. Distilled
models require guidance 1 and no negative prompt. Dimensions must be multiples
of 16, with a minimum of 64; workspace limits may reject large resolutions.
On a 32 GB Mac, start at 256×256. Matrices stay mapped, FFNs and convolutions use
bounded tiles, and the text encoder retains one layer's attention data rather
than a cache for all 32 layers. These are allocation limits, not measured peak
resident-memory or paging guarantees for the full pretrained checkpoint.

Use `gopherllm.OpenImageModel(ctx, directory)` and
`gopherllm.ImageGenerationOptions` to select either image architecture
automatically. `OpenQwenImage21` and `QwenImage21Options` remain compatible.

## Image generation in the Web UI

Start the native server with the downloaded pipeline directory:

```sh
CGO_ENABLED=1 go run ./cmd/gopherllm --serve 127.0.0.1:8080 --chat \
  --image-model-dir .cache/qwen-image21/turbo --threads 8
```

Open `/chat` and choose **Generate image**, or open `/images` directly.
Enter a prompt, select 256×256 or 512×512, and choose a seed. The result appears
in the page with a PNG download. Cancel or leaving the page cancels inference.
The checkpoint determines the schedule (Qwen: eight Turbo or forty base steps;
FLUX: sixteen distilled or fifty base steps).
Start at 256×256 on a 32 GB Mac; larger images cost substantially more work.

The flag does not download weights. It accepts a local pipeline directory with
all three components, including the compact Q8_0 format above. The
FLUX command uses `--image-model-dir .cache/flux3/klein`; the API, cancellation
and model-retention controls are the same for both architectures. The equivalent
JSON configuration is `server.image_model_dir`. An image server does not resume
the last chat model automatically; pass an explicit chat model selector if both
are needed. Image weights are opened lazily. By default they are closed after
each request.
Select **Keep model loaded for the next image** to reuse the loaded model across
successful requests, or choose **Unload model now** to release it immediately.
The choice applies after the next generation; it is initially off on each page
load. Retention avoids reloading weights but holds more memory alongside chat
models. Generation errors, cancellation and server shutdown release the model.
Only one image request is admitted
at a time; another receives HTTP 429. Image requests default to a ten-minute
limit; `--request-timeout` overrides this as well as the chat deadline.

The API is `POST /v1/images/generations`, with JSON such as:

```json
{"prompt":"A red ceramic teapot on a pale blue table.","size":"256x256","seed":42}
```

It returns `created` and `data[0].b64_json` containing a base64 PNG. Optional
`keep_alive: true` retains the loaded model after success; omitted or
false releases it. `POST /v1/images/unload` explicitly unloads it and returns
`{"unloaded":true}`. Unloading during generation receives HTTP 429; cancel the
generation first. The model cache is shared by all clients of this server, so
another client can unload it. Optional `n` must be 1 and `response_format` must be `b64_json`; other fields are rejected.
Prompts are limited to 4096 UTF-8 bytes. Requests cannot choose filesystem paths.
The operator enables the endpoint with `ImageModelDir` in `server.HandlerOptions`
or `server.ServeOptions`; no chat runner is required. Browser-only deployments
disable it because this pipeline runs on the local server, not in WebAssembly.

## German speech

Download `config.json`, `tokenizer_config.json`, `vocab.json` and
`model.safetensors` from
[facebook/mms-tts-deu](https://huggingface.co/facebook/mms-tts-deu/tree/5cbe521869fcf9da4b2b3e85d7810e3005a121dc)
into one directory. The tested revision is
`5cbe521869fcf9da4b2b3e85d7810e3005a121dc`. The weights are approximately
145 MB and licensed **CC-BY-NC 4.0**; the repository's code license does not
override the checkpoint's noncommercial restriction.

```sh
go run ./cmd/tts-synthesize \
  -model /path/to/mms-tts-deu \
  -text 'Hallo Welt. Dies ist GopherLLM.' \
  -out speech.wav -seed 1 -threads 8
```

Use `-text -` to read UTF-8 from stdin. `-speed` accepts 0.25 through 4;
1 preserves the normal speaking rate. `-timeout` defaults to five minutes,
including model loading. Existing output files are never overwritten.

The runtime implements the VITS text encoder, stochastic duration predictor,
inverse flows and HiFi-GAN vocoder. It retains only inference weights, folds
weight normalization once at load time, and uses bounded convolution tiles
instead of materializing full im2col or zero-inserted audio matrices.

Inputs are limited to 512 Unicode characters and predicted audio to 32.768
seconds. Split longer text into sentences. The German character tokenizer
lowercases and removes unsupported characters. It does not verbalize numbers,
dates or abbreviations: write `zwölf` instead of `12`. Other MMS languages,
speaker selection, voice cloning and streaming output are not supported yet.

Embed the same model once and reuse it across utterances:

```go
model, err := gopherllm.OpenTTS(ctx, "/path/to/mms-tts-deu")
if err != nil {
    return err
}
defer model.Close()
audio, err := model.Synthesize(ctx, "Hallo Welt.", gopherllm.TTSOptions{Seed: 1})
if err != nil {
    return err
}
return audio.WriteWAV(writer) // io.Writer; alternatively use audio.Samples
```

Zero `Speed` in the Go API uses the model default. Nil `NoiseScale` and
`DurationNoiseScale` preserve the checkpoint's stochastic defaults; pointers
to zero explicitly disable those sources of variation. A seed is reproducible
within this engine, but does not select PyTorch's random sequence.

## Diffusion images

Download `config.json`, `scheduler_config.json` and
`diffusion_pytorch_model.safetensors` from
[google/ddpm-cifar10-32](https://huggingface.co/google/ddpm-cifar10-32/tree/267b167dc01f0e4e61923ea244e8b988f84deb80)
into one directory. The tested revision is
`267b167dc01f0e4e61923ea244e8b988f84deb80`; weights occupy approximately
143 MB. Consult the checkpoint's model card for its usage terms.

```sh
go run ./cmd/image-generate \
  -model /path/to/ddpm-cifar10-32 \
  -out image.png -seed 42 -steps 50 -threads 8
```

This generates one image without a prompt or class label. The native UNet
uses residual convolution blocks, group normalization, attention and timestep
embeddings. Sampling uses deterministic DDIM (`eta=0`) with the checkpoint's
linear noise schedule and leading timestep spacing. `-steps` accepts 1–1000;
50 is the default. More steps cost more time and do not guarantee a better
image. Existing output files are never overwritten.

```go
model, err := gopherllm.OpenDiffusion(ctx, "/path/to/ddpm-cifar10-32")
if err != nil {
    return err
}
defer model.Close()
img, err := model.Generate(ctx, gopherllm.DiffusionOptions{Seed: 42, Steps: 50})
if err != nil {
    return err
}
return png.Encode(writer, img)
```

The APIs validate model dimensions and reject unsupported architectures.
Each model serializes requests to bound simultaneous workspace use. Context
cancellation applies while waiting and between operators; `Close` waits for
the active request, releases weights and is safe to call again. Returned
audio and images remain valid after subsequent calls or closing the model.

## Validation and next stages

The [speech validation report](../benchmarks/generative/native-tts.md) compares
the encoder, duration predictor, inverse flows and complete waveform against
Transformers using the same nonzero random inputs. The
[diffusion report](../benchmarks/generative/native-diffusion.md) compares UNet
outputs and complete 7-, 23- and 50-step samples against Diffusers. Python is
only needed to regenerate those test references, not for Go inference.

Image editing still needs the vision encoder and condition-image token layout.
Video needs temporal model operations and bounded frame processing. Neither
is implied by text-to-image or CIFAR model support.
