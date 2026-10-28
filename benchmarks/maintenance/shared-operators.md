# Shared operators and workspace cleanup

Measured on 2026-10-25 with Go 1.25.3 on an Apple M2 Max, 32 GiB RAM.
The baseline is commit `c0b4f1c`. Public function signatures and model formats
are unchanged.

## Changes

- TTS and diffusion share a cancellable row-major GEMM fallback. Its worker
  state is reused and cleared after the barrier. Accelerate callers do not
  allocate portable worker state.
- Their convolution patch builders use the same zero-padded window copy.
  Contiguous interiors use `copy`; only padding is cleared. The diffusion
  stride-two gather remains separate.
- Laya and Voxtral share contiguous matrix workspace resizing. Remaining row
  capacity is preserved for BLAS, and replaced slabs lose inactive references.
- Parakeet convolution and attention return their existing projection rows,
  eliminating two temporary row-header arrays per encoder layer.
- TTS reuses exact GELU and finite checks. YOLO uses the identical existing
  `internal/numeric` exponential and sigmoid implementations.
- Shared model metadata, CLI output and transcription audio loaders replace
  duplicate implementations. WAV metadata chunks are skipped without payload
  allocation; PCM is read in bounded blocks. Truncated chunks and partial sample
  frames now return errors.

Batch views with different capacity contracts retain their separate helpers.

## Isolated patch-copy result

`BenchmarkConvolutionPatchCopy` compares the previous per-element bounds check
with the shared copy helper. Both fill a 192-channel, seven-tap, 128-frame
convolution patch, including left and right padding. Inputs and output storage
are allocated before timing. Three 100 ms samples per implementation:

| Implementation | Median | Range | Allocations |
| --- | ---: | ---: | ---: |
| Elementwise checks | 141.216 µs | 135.889–146.339 µs | 0 |
| Padded window copy | 21.045 µs | 20.437–21.139 µs | 0 |

This operation took **85.1% less time (6.71× throughput)**. It is a patch-copy
benchmark, not full synthesis or decoding throughput. The callback dispatch is
present in both benchmark variants. [Raw results](m2-max-patch-copy-2026-10-25.txt).

```sh
go test -run '^$' -bench '^BenchmarkConvolutionPatchCopy$' \
  -benchtime=100ms -count=3 -benchmem ./internal/engine
```

## Whole-model diagnostics and limits

Prebuilt baseline and changed binaries ran in ABBA ABBA order with
`GOMAXPROCS=8`, five timed iterations per sample and no builds or other tests
from this task running concurrently. TTS and diffusion use the actual MMS and
CIFAR checkpoints documented in `../generative/`. Both benchmarks warm the
model before timing. The Parakeet encoder-layer benchmark uses synthetic weights
at the real 1,024-wide / 44-frame dimensions and excludes final LayerNorm.

The host became severely overloaded: `vm.loadavg` reported
`{ 215.56 124.34 113.21 }` and `vm.swapusage` reported 34,304.88 MiB used.
Background applications were not paused. The native UNet samples ranged from
97–244 ms before and 63–214 ms after; the encoder-layer samples also varied by
several multiples. These measurements **do not establish a full-model latency,
token-throughput or paging improvement**. The portable series was stopped after
one baseline and two changed samples when this load was confirmed.

Allocation measurements support the intended structural changes:

- Parakeet's native median falls from 47 to 45 allocations per layer. The two
  eliminated 44-row header arrays account for 2,304 allocated bytes, including
  allocator rounding.
- Native TTS retains approximately 42.34 MB of cumulative allocations per call;
  shared GEMM adds no native worker allocation.
- The three portable diagnostic samples report 7,232 allocations for baseline
  TTS versus 1,566 for each changed run. Warm diffusion reports 125 versus zero.
  These are cumulative allocation counts, not retained memory or process RSS.

[The model samples](m2-max-model-diagnostics-2026-10-25.csv) include the slow
samples and the incomplete portable series. These binaries measure the shared
GEMM, patch-copy and matrix-workspace changes; they precede the validation and
WAV-reader consolidation described below.

Reproduce the model measurements on an otherwise idle host with the local
checkpoint paths set:

```sh
GOMAXPROCS=8 GOPHERLLM_TTS_MODEL=/path/to/mms-tts-deu \
  GOPHERLLM_DIFFUSION_MODEL=/path/to/ddpm-cifar10-32 \
  go test -run '^$' \
  -bench '^(BenchmarkDiffusionCIFARStep|BenchmarkTTSSynthesize|BenchmarkParakeetEncoderLayer)$' \
  -benchtime=5x -benchmem ./internal/engine
```

Repeat with `CGO_ENABLED=0` for the portable implementation. The recorded test
binary SHA-256 values are:

| Backend | Baseline | Changed |
| --- | --- | --- |
| Accelerate | `d1e429e6ef5061a511515e56e9d4f3833cda3195c5a5d0cebd5a8980ca00cee0` | `89d3c8d8e2f0a09c8bdd8a96aabccb4021ab8a30a9949cdccb3d136a4ab34ec2` |
| Portable | `7feca6619510ebf7ecdd13c091a9f4eb20757720f771ed52d830f605a5f66f10` | `675533be3905397e39ab69132a5a1850b3e4fb677b8d95adcabadcb62c44260f` |

## Correctness

Scalar convolution and GEMM comparisons cover padded boundaries, groups,
dilation, transposition, reused/overlapping output buffers and the GEMM tile
tail. Both actual checkpoints pass their independent PyTorch reference tests
with Accelerate and with `CGO_ENABLED=0`, including complete 4/7/23/50-step DDIM
samples and the synthesized waveform. Matrix workspace and Parakeet tests
check storage reuse and input/output separation. Metadata and output-file tests
cover size limits, FIFO rejection, exclusive creation and failed writes.

Validation also passed `go test -tags 'metal coreml' -p=1 ./...`,
`CGO_ENABLED=0 go test -p=1 ./...`, tagged `go vet`, targeted race tests for
the changed operators and file helpers, and portable `go build -p=1 ./...`
for Windows/amd64 and Linux/arm64.

## Shared validation

Audio input, TTS output, decoded TTS/diffusion weights and Laya logits use
`numeric.AllFiniteF32`. Its existing ARM vector
path replaces repeated scalar scans. Laya no longer appends the action value
to its logits just for validation, saving one temporary allocation per question.
The scalar classifier is shared by sampling and model validation as well.

`BenchmarkAudioSampleValidation` compares the old scalar scan with the shared
validator on one and 30 seconds of 16 kHz audio. Three 100 ms samples per case,
Apple M2 Max, `GOMAXPROCS=2`, zero allocations in every case:

| Samples | Scalar median | Shared median |
| --- | ---: | ---: |
| 16,000 | 173.482 µs | 9.758 µs |
| 480,000 | 4.212 ms | 0.168 ms |

These are validation-only measurements on a busy host, not transcription
throughput. Background applications were not paused. The ranges for 480,000
samples were 3.013–4.276 ms and 0.148–0.219 ms respectively.
[Raw measurements](m2-max-audio-validation-2026-10-25.txt).

```sh
GOMAXPROCS=2 go test -run '^$' -bench '^BenchmarkAudioSampleValidation$' \
  -benchtime=100ms -count=3 -benchmem ./internal/engine
```

Server and CLI now share RIFF inspection and PCM conversion in `internal/wav`.
Server-only mono/rate/framing/duration constraints remain at the HTTP boundary.
The CLI rejects duplicate chunks and non-finite float audio; mono negative zero
is preserved, matching the server. Laya also uses the bounded metadata reader
for both configuration files (1 MiB each).

Parakeet, diffusion and Nemotron share the exact division form of SiLU. Laya
and diffusion share their float64-sum softmax phase, retaining their original
maximum-selection passes. Bit-reference tests guard rounding, signed zero and
non-finite values; float32-sum softmax and stable sigmoid variants stay separate.

## Bounded WAV input

The native file loader reads complete PCM frames into a reusable 32 KiB buffer
(or one frame for unusually large channel counts), then converts each block
directly into the final mono output. It no longer retains the entire encoded
recording alongside that output. The HTTP path still decodes its existing
request buffer without an extra copy.

`BenchmarkReadMono` compares the previous whole-chunk allocation with the bounded
reader for 30 seconds of mono audio at 16 kHz. Both use the same PCM conversion
and an in-memory `bytes.Reader`; disk I/O and resampling are excluded. Three
200 ms samples per case, Apple M2 Max, `GOMAXPROCS=2`:

| Input | Whole chunk, median B/op | Bounded, median B/op | Reduction |
| --- | ---: | ---: | ---: |
| PCM16 | 2,891,780 | 1,957,888 | 32.3% |
| Float32 | 3,850,240 | 1,957,888 | 49.1% |

Both paths make two allocations per call. These figures measure allocation
volume, not process RSS or paging. Timings varied substantially, including
slower Float32 samples with bounded reads, so this run establishes no latency
improvement. [Raw results](m2-max-wav-read-2026-10-25.txt).

```sh
GOMAXPROCS=2 go test ./internal/wav -run '^$' -bench '^BenchmarkReadMono$' \
  -benchtime=200ms -count=3 -benchmem
```

Tests compare every output sample bit across block boundaries for mono and
multichannel PCM16/Float32, including negative zero. They also check truncated
final reads, non-finite tail samples, invalid ranges and bounded read sizes.
