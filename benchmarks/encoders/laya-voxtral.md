# Laya and Voxtral encoder scheduling

Measured on 2026-10-25, Apple M2 Max, 32 GiB RAM, macOS 27.2, Go 1.25.3.
Both binaries use Darwin Accelerate with CGO enabled, without Metal or Core ML,
and eight Go workers. The reference is `62fc47f` plus the pending Qwen worker
normalization change, which does not participate in these encoder benchmarks.
Both binaries contain the same benchmark code.

## Changes

- Laya distributes LayerNorm rows across the existing worker pool for batches
  of at least 32 tokens and widths of at least 256. Each row retains its serial
  reduction order. The workspace reuses the dispatch task and clears its tensor
  references after the worker barrier.
- Voxtral's Accelerate attention computes independent heads concurrently.
  Workers have disjoint output columns and separate score matrices; the C call
  waits for all workers before returning. Worker count is bounded by the head
  count, configured threads and a 4 MiB combined score budget. A single score
  matrix larger than that budget runs serially. Single-frame and very small
  attention calls also remain serial.
- The Voxtral encoder writes SiLU-times-up into its gate projection buffer,
  removing a separate FFN activation slab. This also applies to portable builds.

## Timings

Each series ran in reference/changed/changed/reference order, in separate
processes with a 500 ms benchmark target. Builds and tests ran outside the timing
windows. The tables show the median of two measurements per implementation;
background load was not controlled. These are component measurements, not
complete classification or transcription throughput.

| Workload | Reference | Changed | Time reduction |
| --- | ---: | ---: | ---: |
| Laya layer, 128 tokens | 2.532 ms | 2.106 ms | 16.8% |
| Laya layer, 512 tokens | 9.355 ms | 7.389 ms | 21.0% |
| Voxtral offline encoder layer, 372 frames | 41.764 ms | 25.282 ms | 39.5% |
| Voxtral checkpoint layer, 1 streaming frame | 2.257 ms | 2.306 ms | -2.2% |
| Voxtral checkpoint layer, 4 streaming frames | 8.357 ms | 7.513 ms | 10.1% |
| Voxtral checkpoint layer, 16 streaming frames | 8.529 ms | 5.867 ms | 31.2% |

Laya uses synthetic weights with ModernBERT geometry: width 768, gated FFN width
1152, 12 heads of 64 dimensions, and a 64-token local attention window. Its
benchmark includes both normalizations, projections, attention, activation and
residual updates in one layer. A full Laya checkpoint was not available locally.
The two 128-token reference samples were 2.217–2.847 ms and changed samples were
2.094–2.117 ms; the spread limits the precision of the percentage. Both versions
report two allocations per layer.

The offline Voxtral case uses synthetic weights at production dimensions and a
fresh workspace per call. The streaming case loads block zero from
`handy-computer/Voxtral-Mini-4B-Realtime-2602-gguf`, file
`Voxtral-Mini-4B-Realtime-2602-Q6_K.gguf`, and expands that block's projections to
F32 through the existing streaming preparation path. Width is 1280, FFN width
5120, with 32 heads of 64 dimensions. A zero-valued 750-frame history is seeded
before warmup; subsequent calls reuse the workspace and maintain that attention
window. Inputs are deterministic synthetic encoder activations, not recorded
audio. Loading and weight conversion are outside timing.

The one-frame checkpoint samples overlap (reference 2.138–2.376 ms, changed
2.149–2.464 ms) and do not establish an improvement. Parallelizing those heads
helped isolated attention but slightly worsened the complete layer in an earlier
comparison, so that scheduling change was not retained for single frames.

The final isolated attention timings at 4, 16 and 372 frames fell by 71–80%.
The smaller complete-layer gains above include the unchanged projections and
other encoder work. Warm streaming and attention report zero allocations per
operation; occasional pool refills account for nonzero amortized bytes/op.

Raw measurements: [synthetic layers and attention](m2-max-laya-voxtral-2026-10-25.csv),
[checkpoint streaming layer](m2-max-voxtral-checkpoint-2026-10-25.csv).

## Memory

Removing the gate activation copy saves `frames * 5120 * 4` bytes plus one row
header per frame. At 372 frames that is 7.27 MiB, but parallel attention adds
3.17 MiB of score storage (seven matrices instead of one). The net live storage
reduction for those buffers is about **4.11 MiB**. At 16 streaming frames the
extra attention scores exceed the removed FFN slab by about **14.75 KiB**; at
one frame the serial attention path saves about 20 KiB from the removed slab.
No additional weight copies are introduced.

These are buffer sizes, not RSS or paging measurements. The existing score pool
can retain a larger previous allocation, and concurrent calls need independent
scratch. The 4 MiB limit bounds a call's parallel score requirement, not total
process memory. Laya's embedded tasks add 624 bytes per workspace on arm64.

## Reproduction and validation

Build reference and changed sources into separate test binaries, including the
same added benchmarks in the reference:

```sh
CGO_ENABLED=1 go test -c -o /tmp/encoders.test ./internal/engine
GOMAXPROCS=8 /tmp/encoders.test -test.run '^$' \
  -test.bench '^(BenchmarkLayaLayer|BenchmarkVoxtralAttention|BenchmarkVoxtralEncoderLayer)$' \
  -test.benchtime=500ms
GOMAXPROCS=8 GOPHERLLM_VOXTRAL_MODEL=/path/to/Voxtral-Mini-4B-Realtime-2602-Q6_K.gguf \
  /tmp/encoders.test -test.run '^TestVoxtralEncoderWorkspaceRealCheckpoint$' \
  -test.bench '^BenchmarkVoxtralEncoderStreamCheckpoint$/rows_reused/' \
  -test.benchtime=500ms
```

Alternate the binaries in the order above. Laya normalization tests compare
every output bit against the original serial row operation. Voxtral tests cover
gate aliasing, guards, different window and head counts, padded KV strides,
concurrent attention calls, workspace reuse and the local checkpoint. The
parallel attention reference uses the same original serial CBLAS/softmax order.

Portable and Metal/Core ML test suites, targeted Laya/Voxtral race tests,
native `go vet` and all six `make cross-build` targets passed.
