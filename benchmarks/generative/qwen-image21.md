# Native Qwen-Image-2.1 validation

The Go pipeline implements text-to-image inference with the Qwen3-VL text
encoder, Qwen-Image-2.1 DiT, Flow Matching Euler scheduler and RGBA decoder.
Tests below ran on an Apple M2 Max with 32 GiB RAM and Go 1.25.3. The runtime
uses Go/SIMD and Accelerate; it does not invoke Python or a model server.
Metal and Core ML do not yet execute this image graph.

## Sources and checkpoint format

- [Turbo checkpoint](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/tree/fc13ba1986fcb905854dc8de2f52eee395f49b5b):
  `fc13ba1986fcb905854dc8de2f52eee395f49b5b`.
- [Base checkpoint](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d26bb61231c349cf6b7896fa83353113880e1ba3):
  `d26bb61231c349cf6b7896fa83353113880e1ba3`.
- Reference implementations: Transformers 4.57.1 and Diffusers 0.41.0,
  PyTorch 2.8.0, CPU F32. Fixture manifests record source hashes.

The offline importer reads bounded HTTP ranges and converts matrix weights
directly into Q8_0 GGUF components. It keeps normalization and VAE tensors in
their source precision and omits unused text-encoder vision and LM-head weights.
Each completed shard has a SHA-256 record; interrupted conversions resume at
the last complete tensor after checking checkpoint identity and layout.
Model weights retain their upstream license, independently of the Go code.
The compact Turbo components occupy approximately 15.159 GiB: text encoder
7.489 GiB, transformer 7.041 GiB and VAE 0.629 GiB, plus small metadata files.

## Independent numerical tests

Fixtures use small, randomly initialized official components, including BF16
VAE weights. They test equations and tensor layout, not pretrained image
quality. Initial noise is stored explicitly because Go and PyTorch RNG streams
are different. Runtime inference consumes neither Python nor these references.

| Comparison | Result |
| --- | --- |
| Text conditioning before final RMS norm, 48 token rows | Maximum absolute error 7.45e-9 |
| Complete tokenizer → encoder → Turbo8 DiT/Euler → VAE, 64×32 | One RGBA channel differs by one byte value |
| Same complete pipeline, 64×64 | Byte-identical RGBA pixels |
| DiT packed Q8_0 versus the same Q8_0 weights expanded to F32 | Passed numerical tolerance |
| VAE direct and overlapping tiled decoding | All RGBA channels within one byte value |

Both portable Go and native `metal coreml` builds pass these tests. The latter
also checks compatibility with those optional backends; it does not mean that
the Qwen image graph runs on the GPU or Neural Engine.

Additional tests cover the saved Turbo sigma sequence, base scheduler shifting,
text-prefix KV reuse, three-axis RoPE, cancellation, malformed tensor storage,
mapped-weight lifetime and bounded scalar expansion. The official tokenizer's
Unicode, digit, whitespace and special-token behavior has separate fixtures.

## Pretrained Turbo generation

On 2026-10-25 the complete pinned Turbo checkpoint was downloaded, converted
and verified against the generated shard checksums. Two separate native CLI
processes generated images with seed 42, guidance 1 and the saved eight-step
schedule. Both used the prompt:

> A red ceramic teapot on a pale blue table, studio photograph.

| Image | Process wall time | Maximum RSS | macOS peak memory footprint |
| --- | ---: | ---: | ---: |
| 64×64 smoke test | 30.39 s | 13.52 GiB | 1.14 GiB |
| 256×256 generation | 92.93 s | 13.18 GiB | 1.97 GiB |

The resulting [256×256 PNG](qwen-image21-turbo-teapot.png) was decoded and
visually inspected: it shows a red ceramic teapot with a spout, lid and handle
on a pale blue surface and background. Its SHA-256 is
`1f5ad1ff44bc63a9ec4fd72adeb25ec9e18d1be4fe119a5b3200fae4c7fa89ab`.
This verifies an actual pretrained text-to-image run, beyond the small numerical
fixtures. It is not a broad image-quality evaluation or a BF16/Q8 comparison.

![Native Turbo output, seed 42](qwen-image21-turbo-teapot.png)

Times and process counters come from `/usr/bin/time -l` and include model
loading, generation and PNG output. The filesystem cache was warm from the
preceding checksum verification. Each size has one measured run, with eight
Go workers and `GOMAXPROCS=8`; there is no performance baseline. RSS includes
resident mapped weights, whereas macOS's footprint accounting treats clean
file-backed pages differently. The footprint column must not be read as the
total RAM needed to keep the model resident.

The system already reported approximately 15.6 GiB of used swap before these
runs. Process page-fault and swap counters are preserved without translating
them into disk traffic or claiming paging improvements. See the
[complete measurement record](qwen-image21-turbo-m2-max.json), including binary
and shard hashes. The base checkpoint, larger images and other Macs have not
had a pretrained generation test in this change.

## Projection batching comparison

On 2026-10-26, four fresh CLI processes ran the same 256×256 Turbo workload
in before/after/after/before order. Model loading and PNG output are included;
the filesystem cache was warm. No builds or test suites ran during measurement.

| Variant | Wall time, both runs | Median | Peak footprint, both runs |
| --- | --- | ---: | --- |
| Original 128-token slabs | 97.39 s, 96.52 s | 96.96 s | 2,223 / 2,222 MB |
| Balanced slabs, larger packed-weight GEMMs | 72.16 s, 67.46 s | 69.81 s | 2,233 / 2,232 MB |

The median latency fell **28.0%**, or a **1.39× speedup**, for this prompt and
machine. Two runs per variant do not establish a confidence interval. The
optimization balances short tails so they remain eligible for Accelerate,
uses up to 256 token rows per projection/FFN slab, and fills more of the
existing 4 MiB weight-expansion budget per GEMM. Projection inputs remain
bounded at 8 MiB; weights remain Q8_0 and mmap-backed. The FFN slab grows by
12 MiB at the official model's width, so this is a latency improvement with a
small workspace tradeoff, not a claim of reduced memory use.

The [optimized PNG](qwen-image21-turbo-teapot-optimized.png) was inspected and
compared with the original: 20 of 262,144 RGBA channel values differ by one
byte value; all others match. Both runs within each variant produced identical
PNG files. Matrix blocking changes floating-point rounding slightly.

Maximum RSS ranged from 11.42–12.04 GiB before and 11.67–12.25 GiB after.
Mapped-page residency and system swap were not controlled; these measurements
do not show a paging reduction. The JSON record includes all four runs,
binary hashes, image hashes and process counters. Larger images and other
Macs need separate performance measurements.

## Memory boundaries

Large transformer matrices remain mmap-backed. Projection and feed-forward
work use at most 256 token rows per slab, with an 8 MiB projection-input
budget. The DiT retains only the prompt's
prefix K/V between steps. Its estimated state is limited to 2 GiB per guidance
branch before allocation. The text encoder uses 32-token prefill chunks and
bounds its prompt KV allocation at 2 GiB. The VAE expands its used decoder
weights once, closes its source mapping and decodes images above 256 pixels
in overlapping tiles with one retained tile row. These are implementation
limits, not measurements of process RSS or system paging.

The API defaults to 256×256. Supporting dimensions up to 2048 does not establish
practical 2K latency or paging behavior on a 32 GiB machine. Q8_0 introduces
quantization error and is not bitwise equivalent to the original BF16 model.

## Reproduction

```sh
python scripts/download_qwen_image21.py .cache/qwen-image21/turbo --variant turbo
CGO_ENABLED=1 go build -o /tmp/image-generate ./cmd/image-generate
GOMAXPROCS=8 /usr/bin/time -l /tmp/image-generate \
  -model .cache/qwen-image21/turbo \
  -prompt 'A red ceramic teapot on a pale blue table, studio photograph.' \
  -width 256 -height 256 -seed 42 -threads 8 -out /tmp/teapot.png

go test -run 'QwenImage21|QI21' -v ./internal/engine
CGO_ENABLED=0 go test -run 'QwenImage21|QI21' ./internal/engine
python -m unittest discover -s scripts -p 'test_download_qwen_image21.py'

# Synthetic packed projections, without a checkpoint download.
GOMAXPROCS=8 go test -run '^$' -bench '^BenchmarkQwenImage21Projection$' \
  -benchtime=3x -count=2 ./internal/engine
# Optional full-graph profile; loading is outside benchmark timing.
GOPHERLLM_QWEN_IMAGE_MODEL="$PWD/.cache/qwen-image21/turbo" GOMAXPROCS=8 \
  go test -run '^$' -bench '^BenchmarkQwenImage21Generate$' -benchtime=1x ./internal/engine
```

The reference generators in `scripts/` reproduce the encoder, transformer,
VAE and complete pipeline fixtures. Image editing, reference-image conditions,
LoRA adapters and video are outside the implemented scope.

## Batched prefix/image attention

On 2026-10-26, the DiT attention path was changed from per-query online
attention to bounded Accelerate QKᵀ/softmax/PV tiles. Heads run concurrently;
Q/K/V stay in token-major storage, and the combined score workspace is capped
at 4 MiB. Prefix queries remain causal and image queries retain bidirectional
attention. Cached prefix K/V and portable fallback behavior are unchanged.

Synthetic attention benchmark on the M2 Max (Go 1.25.3, GOMAXPROCS=12,
32 heads × 128 dimensions, 256 image tokens plus 32 prefix tokens, three runs
of ten iterations; medians):

| Queries | Online attention | Batched attention | Batched allocations |
| --- | ---: | ---: | ---: |
| Prefix + image | 19.92 ms | 2.44 ms | 0 B/op |
| Image, prefix cached | 29.65 ms | 2.04 ms | 0 B/op |

The kernel is faster, but the complete model comparison does **not** establish
a wall-time speedup. Four fresh CLI processes used the same pretrained prompt,
256×256, seed 42, Turbo eight steps and eight threads in ABBA order:

| Variant | Wall time | User CPU time | Page faults |
| --- | --- | --- | --- |
| Before | 120.29 / 137.47 s | 157.80 / 161.99 s | 2.55 / 3.09 million |
| Batched attention | 110.60 / 150.54 s | 127.74 / 132.20 s | 2.57 / 4.16 million |

Median wall times were 128.88 and 130.57 seconds. Background work and paging
were uncontrolled despite freeing RAM; these runs cannot be compared directly
to the earlier 69.81-second projection result. The reduced CPU work does not
remove the cost of streaming the model under memory pressure. Peak process
footprint was 2.20–2.32 GB across the four runs; this is not total system memory
or the size of the mapped checkpoint.

Both optimized runs produced identical PNGs. Against the original, 54 of
262,144 RGBA channels differ by one byte value (maximum difference 1).
Tests additionally compare masks, cached-prefix queries, cancellation and
large-key score bounds against the online path and existing reference fixtures.

Raw measurements and image hashes: [attention results](qwen-image21-attention-m2-max.json).
Reproduce the kernel comparison with:

```sh
GOMAXPROCS=12 CGO_ENABLED=1 go test -run '^$' \
  -bench '^BenchmarkQwenImage21Attention$' -benchtime=10x -count=3 ./internal/engine
```

## Shared decoder and projection changes, 2026-10-27

Qwen now shares pooled projection row views and fused nearest-neighbor
upsampling/convolution with FLUX. The pool drops tensor references before
returning; upsampling expands only a bounded row window. In the synthetic
256-channel, 128×128 → 256×256 convolution benchmark, allocated bytes fall
from 128 to 64 MiB. This is not a full-model RSS or paging measurement.
Existing Qwen reference images remain within their original tolerances.
See the [shared kernel measurements](flux3.md#kernel-and-allocation-changes-2026-10-27)
for raw data, concurrent-load limitations and reproduction commands.

On 2026-10-28, residual blocks also began reusing the consumed normalization
buffer for their second convolution's output. This removes an intermediate
feature-map allocation in both VAEs; Qwen's existing reference images still
pass. The [FLUX fixture comparison](flux3.md#residual-workspace-reuse-2026-10-28)
measures this change, without a new pretrained Qwen throughput claim.
