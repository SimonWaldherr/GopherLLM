# FLUX 3 Image Klein validation

Status on 2026-10-28: native text-to-image implementation, **experimental**.
The distilled pretrained checkpoint now passes a native 256×256 generation test
on the 32 GiB M2 Max. This is one prompt/seed with Q8_0 matrices, not a quality
evaluation or a validation of larger images, base CFG or editing.

## Pretrained generation, 2026-10-28

The authenticated range importer downloaded the pinned distilled checkpoint
and converted its required matrices to Q8_0. The four component GGUF files
occupy 8.59 GiB; text encoder layers 33–36 and unused image branches are omitted.
Go 1.27.3 with CGO/Accelerate ran the native engine from commit `d440e6f`, with
GOMAXPROCS=8, eight workers, 16 steps and seed 42. Python was used for download
and measurement only; the generator itself was the compiled Go executable.

| Metric | Observed value |
| --- | ---: |
| Output | 256×256 RGB PNG |
| Wall time, including loading and process teardown | 103.21 s |
| Maximum resident set size, reported by macOS `time -l` | 7,396,179,968 bytes (6.89 GiB) |
| Process page faults | 525,969 |
| System swap-ins / swap-outs during the run | 10,413 / 58,028 pages |

The image visibly matches the prompt's single red teapot, left handle, right
spout, pale blue table and cream background. This is a visual smoke test;
there is no full-checkpoint BF16/differential image comparison. Loading is
included but the OS file cache was not cleared after downloading. This is a
single observation without a controlled background load or speedup comparison.

The process reports zero `swaps`, while system-wide swap counters increased.
Those counters include other applications and cannot attribute paging solely
to GopherLLM. RSS includes resident mapped pages; it is not a total-system RAM
or swap-free-operation guarantee.

![Native FLUX 256×256 teapot](flux3-teapot-256.png)

The complete prompt, component/image hashes, VM snapshots and measurements are
in [`flux3-pretrained-256-m2-max.json`](flux3-pretrained-256-m2-max.json).

## Reference and scope

- [BFL implementation](https://github.com/black-forest-labs/flux-3-image-klein/tree/bfc8889a4497f846416f253a3972a1941ac698b9),
  revision `bfc8889a4497f846416f253a3972a1941ac698b9`.
- [Distilled weights](https://huggingface.co/black-forest-labs/flux-3-image-klein/tree/337337621b8526e120f2275d250242f4a6bedd75),
  pinned for the authenticated range importer.
- Qwen3-VL-4B has 36 decoder layers. FLUX consumes hidden states 4, 8, …, 32,
  so the last four layers and final RMSNorm are omitted. The reference fixture
  retains all 36 layers to verify that intermediate features are selected.
- Text-to-image only. The base path uses CFG; the distilled path conditions on
  the current and target times. BFL's alpha-16 schedule defaults to 50 and 16
  steps respectively. No prompt upsampler, image editing, layout conditioning,
  LoRA, FP8/NVFP4 CUDA kernels or Diffusers tensor-name conversion is included.

`scripts/flux3-reference.py` runs BFL's unmodified module implementations with
smaller dimension constants. Fixtures use PyTorch 2.8.0 and Transformers 4.57.1
in float32 with synthetic parameters, not pretrained model weights. Source
hashes are recorded in `testdata/flux3/reference.json` and verified before the
generator imports the reference modules. VAE parameter patterns
are periodic and nonzero so the fixture compresses well; normalization and
latent statistics are nonuniform. The latent channels, GroupNorm groups and
decoder resampling/residual structure match the production architecture.

## Checks completed

| Check | Result |
| --- | --- |
| Text conditioning, four prompts including padding/Unicode | Maximum allowed absolute difference `3e-5` from the reference |
| DiT, base and flow-map, two rectangular grids and two time pairs | Maximum allowed absolute difference `3e-5`; repeated calls agree |
| Euler schedules at 1, 3, 16 and 50 steps | Maximum allowed absolute difference `2e-7` |
| Complete two-step pipelines at 64×64 and 96×64, including base CFG | All RGB channels within one 8-bit value; alpha is 255 |
| Native Q8_0 component loading | Both variants load, omit unused text layers and retain packed matrix storage |
| Cancellation, close and validation | Invalid inputs rejected; cancellation after the first step permits later reuse; closed models reject generation |
| CLI smoke test | A 64×96 PNG generated from the synthetic fixture |
| Pretrained distilled CLI | Native 256×256, 16-step Q8_0 generation completes and produces a plausible prompt-matching image |
| HTTP smoke test | `/images`, two 256×256 native generations with retention on/off, and `/v1/images/unload` passed using the synthetic fixture |
| Existing Qwen image tests | Pass with shared projection/attention infrastructure and automatic model dispatch |
| Full `go test ./...` | Pass with local loopback test servers permitted |
| Targeted `-race` tests | FLUX, Qwen image, factory and image server tests pass |
| Portable build (`CGO_ENABLED=0`) | Targeted engine, server and CLI tests pass |
| `metal coreml` build tags | Targeted engine/server/CLI tests pass; image inference itself uses Accelerate/Go |
| Windows amd64 | Root test binary cross-compiles; not executed on Windows |
| Release CLI cross-build | Darwin, Linux and Windows binaries build for amd64 and arm64; foreign binaries are not executed |
| Native Apple vet | Changed engine/server/CLI packages pass with `metal coreml` tags |
| Offline download tests | 14 tests pass, including credential stripping on CDN redirects |
| Web image controls | Four JavaScript tests pass |

The tiny fixture's latency is deliberately not reported as model performance.
An HTTP smoke test verifies the route through the real native engine; it does
not verify pretrained prompt adherence.

## Kernel and allocation changes, 2026-10-27

The encoder now batches causal GQA through Accelerate without expanding shared
K/V heads. The same bounded attention implementation serves FLUX and Qwen image
transformers; the portable path retains online attention. Text RoPE angles are
prepared once per prompt and shared across Q/K and all 32 encoder layers.

Projection row views are pooled and cleared after the worker barrier so they
cannot retain prompt/image tensors. FLUX reuses the completed Q projection's
buffer for attention output and compacts SwiGLU activations in place. Both
image decoders fuse nearest-neighbor upsampling into convolution: only a small
row window is expanded, then reused for all horizontal kernel offsets.

Measurements used an M2 Max with 32 GiB RAM, Go 1.25.3 and Accelerate. Five
paired repetitions alternated reference/optimized execution order. Builds,
other benchmarks and simulators were active; **wall times are noisy**. Raw
samples, binary hashes and conditions are in
[`flux3-optimization-m2-max.json`](flux3-optimization-m2-max.json).

| Allocation benchmark, 8 Go workers | Before | After |
| --- | ---: | ---: |
| Synthetic FLUX encoder, bytes/call | 3,221,546 | 800,120 |
| Synthetic FLUX DiT, bytes/warm step | 802,688 | 6,272 |
| Synthetic FLUX VAE, bytes/256×256 decode | 228,893,296 | 201,631,517 |
| Upsample + convolution, 256 channels, 128×128 → 256×256 | 128 MiB | 64 MiB |

These are median **allocated bytes**, not peak RSS or paging measurements.
The upsample kernel uses production-sized tensors; the graph tests use the
small reference fixture with synthetic weights. For production FLUX dimensions,
the removed DiT slabs save a calculated 13.875 MiB per state at 256×256, or
22.875 MiB at 512×512, with 160 text tokens. Base CFG owns two states.

For GQA with 32 query heads, 8 KV heads, head width 128, 160 keys and one extra
padding query, the 8-worker median fell from **5.003 ms to 0.995 ms**. Ranges were
4.640–18.620 ms and 0.805–1.454 ms respectively. This supports the kernel change,
but the concurrent load prevents a stable end-to-end speedup claim. Longer
attention tests and the small graph timings are retained in the raw results
as diagnostics, not advertised as pretrained model performance.

Correctness checks cover independent GQA softmax/indexing, causal/padding masks,
tile boundaries, in-place SwiGLU, concurrent projection calls, cancellation,
and fused convolution versus a materialized nearest-neighbor reference.
The existing BFL and Qwen reference-feature/image tolerances are unchanged.

## Residual workspace reuse, 2026-10-28

Both VAEs now write each residual block's second convolution into its consumed
normalization buffer. The input and shortcut remain separate, and all readers
finish before reuse. A larger channel count grows the destination as needed;
dirty destination tests cover both 1×1 and tiled 3×3 convolutions, including
cancellation. No cross-request activation cache is added.

Five alternating before/after pairs on the same M2 Max used the synthetic
256×256 RGB decoder graph, with three decodes per sample and GOMAXPROCS=8:

| Metric, median | Before | After |
| --- | ---: | ---: |
| Allocated bytes per image | 201,631,352 | 154,967,408 |
| Decode time | 206.40 ms | 202.39 ms |

Allocated bytes fell about 23%. The small timing difference is insufficient
to establish a throughput improvement, given uncontrolled background work.
The fixture has fewer channels than the pretrained VAE; these are not full
checkpoint RSS or paging measurements. Raw samples and binary hashes are in
[`flux3-vae-reuse-m2-max.json`](flux3-vae-reuse-m2-max.json).

## Memory behavior

Matrix weights remain mmap-backed. Text attention stores only the active layer,
and identical right-padding queries are computed once before replicating their
features. Static text projection is reused across denoising steps. DiT FFNs,
attention score tiles and VAE convolutions use bounded workspaces. VAE decoder
weights are owned float32 arrays; their source mapping is closed after loading.

The text workspace is limited to 1 GiB; individual DiT states and the estimated
VAE workspace are limited to 2 GiB. These limits exclude mapped model files,
allocator overhead and other applications. Base CFG uses two DiT states.
The estimates are allocation guards, not a guarantee of swap-free operation.

## VAE attention reuse, 2026-10-29

The decoder now keeps four channel maps instead of nine during attention.
Consumed normalization and projection buffers are reused only after their
readers have finished. Q/K/V layout copies share a tiled transpose helper;
GEMM shapes, attention backend and softmax arithmetic stay the same.

On M2 Max, a 512-channel 32×32 attention call measured **10.009 → 7.590 ms**
(1.32×), with allocated bytes falling from **19 to 9 MiB**. These are
per-call operator measurements with synthetic weights and a fresh workspace,
not peak model RSS or complete image-generation throughput.
[Raw samples and allocation counts](../maintenance/vae-attention-reuse-m2-max-2026-10-29.json).

A subsequent pretrained 256×256, 16-step run produced the exact reference PNG
hash `3d0ae815b40c8c45ed82c7086e1c94b829d3ab2c6a7f204b39ef4ac65083fca7`.
The run took 210.45 seconds including load and reached 8.12 GiB process peak
RSS on the busy host. This is a correctness check, not a controlled throughput
or paging comparison. [Validation record](flux3-pretrained-256-validation-2026-10-29.json).

```sh
CGO_ENABLED=1 GOMAXPROCS=8 go test ./internal/engine \
  -run 'TestVAEAttentionReuse|TestTransposeVAEInto' \
  -bench '^BenchmarkVAEAttention' -benchtime=7x -count=1 -benchmem
```

## Reproduction

Place `dit.py`, `autoencoder.py`, `text_encoder.py` and `sampling.py` from the
pinned BFL revision in a local reference directory, then run:

```sh
python scripts/flux3-reference.py /path/to/reference testdata/flux3
go test ./internal/engine -run 'TestFlux3|TestOpenImageModel' -count=1
GOMAXPROCS=8 go test ./internal/engine -run '^$' -bench 'BenchmarkFlux3TextAttention|BenchmarkImageUpsampleConv' -benchmem -count=5
GOMAXPROCS=8 go test ./internal/engine -run '^$' -bench BenchmarkFlux3Stages -benchmem -benchtime=3x -count=5
CGO_ENABLED=0 go test ./internal/engine -run 'TestFlux3|TestOpenImageModel' -count=1
go test -race ./internal/engine ./server -run 'TestFlux3|TestOpenImageModel|TestQwenImage21|TestImage' -count=1
python -m unittest discover -s scripts -p 'test_download_*.py'
node --test server/web_ui/images_test.cjs
```

To reproduce the pretrained test, obtain model access and authenticate locally,
then follow [the download and generation instructions](../../docs/generation.md#flux-3-image-klein)
with the prompt and settings in the pretrained measurement JSON.
