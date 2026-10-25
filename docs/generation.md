# Native speech and image generation

The first generation stage exposes two local Go APIs and standalone commands:

| Capability | Supported checkpoint | Output |
| --- | --- | --- |
| German text-to-speech | `facebook/mms-tts-deu` | Mono 16 kHz PCM/WAV |
| Unconditional image generation | `google/ddpm-cifar10-32` | 32×32 RGB/PNG |

Inference runs in the GopherLLM process. Neither command starts Python or
downloads weights. Matrix operations use the existing Accelerate backend on
macOS with CGO and Go/SIMD kernels otherwise. These paths do not yet use Metal
or Core ML. The Go module still has no third-party dependencies.

Text-conditioned image generation and video generation are not implemented.
The image model below is a small diffusion reference model, not a replacement
for a text-to-image model. These first APIs are not yet exposed through the
chat server, mobile package or language bindings.

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

Both APIs validate model dimensions and reject unsupported architectures.
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

The next image stage needs a text encoder, conditioned denoiser and image
autoencoder with a separate real-checkpoint reference test. Video then needs
temporal model operations and bounded frame processing. Neither is implied
by the current CIFAR model support.
