# Native German MMS speech synthesis

Validated on 2026-10-25 on an Apple M2 Max with 32 GiB RAM. The implementation
loads `facebook/mms-tts-deu` directly and produces mono PCM at 16 kHz. Text
encoding, stochastic duration prediction, reverse residual flows and HiFiGAN
run in Go. Darwin with CGO uses the existing Accelerate GEMM; the portable
build uses the existing Go/SIMD kernels. Python is only a reference generator.

## Checkpoint and reference

- Model revision: `5cbe521869fcf9da4b2b3e85d7810e3005a121dc`.
- `model.safetensors`: 145,232,888 bytes.
- SHA-256: `6068054cb03caf5a00902e17f083e042d247aa4ce8919d6a93567f5262c80ec5`.
- Reference: PyTorch 2.8.0, Transformers 4.57.1, safetensors 0.6.2, CPU F32.
- Text: `Hallo Welt.`; PyTorch seed 1337; checkpoint noise scales 0.667 and 0.8.
- Output: 16,640 PCM samples, 1.04 seconds.

The fixture records both Gaussian noise inputs and every major graph boundary.
Go receives the same noise arrays, so the comparison does not assume matching
random generators. The reference exercises nonzero noise and the full inverse
rational-quadratic duration spline. Runtime tests also check repeatability of
Go's own seeded generator and the explicitly requested zero-noise option.

The German tokenizer uses its original 45-character vocabulary, lowercase
normalization, removal of unsupported characters and blank insertion. It does
not need a phonemizer, Python or an external service. Unit cases cover umlauts,
`ß`, punctuation and newlines as well as the full reference sentence.

Sources: [pinned model](https://huggingface.co/facebook/mms-tts-deu/tree/5cbe521869fcf9da4b2b3e85d7810e3005a121dc),
[VITS reference](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/vits/modeling_vits.py),
[tokenizer reference](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/vits/tokenization_vits.py).
The checkpoint is licensed CC-BY-NC-4.0. Model weights are not distributed in
this repository. `testdata/tts_mms.safetensors` contains only reference inputs
and outputs; `testdata/tts_vocoder.safetensors` uses synthetic random weights.

## Numerical results

Accelerate build, maximum absolute error and relative RMS error over every
value at the indicated graph boundary:

| Boundary | Maximum absolute error | Relative RMS error |
| --- | ---: | ---: |
| Text encoder | 9.54e-7 | 1.82e-7 |
| Prior means | 2.03e-6 | 4.32e-7 |
| Prior log standard deviations | 3.58e-7 | 1.64e-7 |
| Stochastic log durations | 3.10e-6 | 1.44e-6 |
| Expanded sampled prior | 2.03e-6 | 2.19e-7 |
| Reverse residual flows | 9.54e-6 | 6.26e-7 |
| Complete PCM waveform | 4.39e-5 | 3.15e-5 |

The full reference also passes with `CGO_ENABLED=0`: waveform maximum error
1.14e-5 and relative RMS error 7.71e-6. This includes the text encoder, sampled
durations and inverse flows, as well as repeated public-API synthesis calls.

An independent seven-frame vocoder input produced 1,792 samples with maximum
error 2.32e-6 and RMSE 2.31e-7. Its portable build also passed, with maximum
error 2.15e-6. A small checked-in vocoder fixture runs without downloaded model
weights. Convolution tests exercise grouping, dilation, transposed convolution,
tile boundaries, malformed dimensions and cancellation.

## Memory and latency

The loader retains 460 inference tensors containing 28,274,395 F32 values:
113,097,580 bytes (107.86 MiB). Training-only posterior weights and the duration
flow unused during reverse inference are omitted. WaveNet weight normalization
is folded once. The checkpoint mapping is closed before `OpenTTS` returns.
These are resident weight-buffer sizes, not process RSS.

Convolutions materialize at most 128 output frames per im2col tile. Transposed
convolutions scatter projected tiles without allocating a zero-expanded input.
Synthesis calls on one model are serialized; each utterance has a 512-character
input bound and a 2,048-frame output bound (32.768 seconds). Longer text must be
split into sentences. The text encoder and flows allocate their own
intermediates. The vocoder reuses six activation buffers and one set of
convolution tiles within each utterance.
Residual-input activation is computed once into a reusable buffer, preserving
the original input for its later addition. These buffers become reclaimable
when the synthesis call finishes; they are not kept in the model or a global
pool. The returned PCM has its own storage.

A paired warm synthesis benchmark generated 1.232 seconds of audio per call
from `Hallo Welt.` with seed 19. Model loading was excluded. Four baseline and
four optimized runs were interleaved in ABBAABBA order, with three iterations
per run. Both binaries used `GOMAXPROCS=8` and eight workers:

| Metric | Baseline | Reused vocoder buffers |
| --- | ---: | ---: |
| Median synthesis time | 373.44 ms | 371.93 ms |
| Observed time range | 364–384 ms | 357–376 ms |
| Median allocated bytes per call | 286,932,138 | 42,338,714 |
| Median allocations per call | 1,744 | 1,471.5 |

The optimization reduces allocated bytes by 85.2%. The timing ranges overlap;
the measurements do not establish a synthesis speedup. Allocated bytes are the
cumulative allocation volume per request, not peak live memory or process RSS.
[Raw kernel measurements](m2-max-native-kernels-2026-10-25.csv).

Four separate CLI processes ran in ABBA order with eight workers, seed 1 and
the sentence `Hallo Welt. Dies ist ein Test der Sprachausgabe.`. Each produced
5.184 seconds of audio. The table reports medians of two runs per version;
maximum RSS was collected with `/usr/bin/time -l` for the complete process,
including model loading and WAV writing:

| Metric | Baseline | Reused vocoder buffers |
| --- | ---: | ---: |
| Synthesis time | 1.5705 s | 1.5325 s |
| Maximum RSS | 363,913,216 bytes (347.05 MiB) | 245,350,400 bytes (233.98 MiB) |

Maximum RSS decreased by 32.6%. Two runs per version are insufficient to infer
a latency improvement. The first baseline process incurred 6,973 page faults;
the other processes incurred 25–57, reflecting different file-cache states.
All reported zero swaps. These observations do not establish reduced paging.
[Raw CLI measurements](m2-max-native-cli-2026-10-25.csv).

All four WAV files were byte-identical, SHA-256
`e82dffeb34c04d6dc22481ac0a58801f745a71f5adbcb6f1274ffbc2ac133d8f`.
The baseline was the uncommitted native TTS implementation immediately before
this optimization, not a released version or Git commit. The benchmark binary
SHA-256 values were:

- Baseline: `fa2fdda1f57a9ed263150eb36f2ec6d58dda3d11260eb5dd9fa75a10134dead6`.
- Optimized: `3f1b03d71e97defadbf04596bce8493bfda746aba9e5a0abf723aacbebd3ff5f`.

The generated WAV was also passed to GopherLLM's native Parakeet-TDT v3
recognizer with Q8_0 weights. Its transcript was `Hallo Welt! Dies ist ein Test
der Sprachausgabe.`. This is a single intelligibility smoke test, not a speech
quality or word-error-rate evaluation.

## Reproduction

Place `config.json`, `tokenizer_config.json`, `vocab.json` and
`model.safetensors` from the revision above in a local directory. No test
downloads model weights or invokes Python:

```sh
go test -run '^TestTTS' ./internal/engine

GOPHERLLM_TTS_MODEL=/path/to/mms-tts-deu \
  go test -run '^TestTTSMMSReference$' -count=1 -v ./internal/engine

GOMAXPROCS=8 GOPHERLLM_TTS_MODEL=/path/to/mms-tts-deu \
  go test -run '^$' -bench '^BenchmarkTTSSynthesize$' -benchtime=3x -benchmem ./internal/engine

go build -o tts-synthesize ./cmd/tts-synthesize
GOMAXPROCS=8 /usr/bin/time -l ./tts-synthesize -model /path/to/mms-tts-deu \
  -text 'Hallo Welt. Dies ist ein Test der Sprachausgabe.' \
  -threads 8 -seed 1 -out speech.wav
```

Regenerate the full reference with Python 3.11 or newer (for
`hashlib.file_digest`) and the pinned Python packages in an isolated development
environment:

```sh
python scripts/tts-reference.py --model /path/to/mms-tts-deu \
  --output testdata/tts_mms.safetensors
```

`GOPHERLLM_TTS_REFERENCE` may point to another generated reference file. Go's
seed is deliberately not expected to match PyTorch's seed; input noise is
captured in the fixture for numerical comparisons.

## Short pointwise projections, 2026-10-29

Unpadded 1×1 convolutions with at most four output channels per group and
128 frames now write GEMM results directly to the caller's output. Other
shapes keep the tiled packing path. Input/output overlap checks and the
long-utterance cancellation boundaries remain in place.

For a 192→2 projection over 64 frames, nine alternating batches on M2 Max
measured **1.42×** portable and **1.20×** Accelerate throughput. Packing
scratch falls from **49,664 bytes to zero** with the output already allocated.
These are operator measurements, not whole-utterance synthesis timings.
[All cases, raw output and limits](../maintenance/tts-pointwise-m2-max-2026-10-29.json).

```sh
CGO_ENABLED=0 GOMAXPROCS=8 go test ./internal/engine -run '^TestTTSPointwiseDirect' \
  -bench '^BenchmarkTTSPointwiseDirect$' -benchtime=9x -count=1 -benchmem
```

Repeat with `CGO_ENABLED=1` on macOS to measure the Accelerate backend.
