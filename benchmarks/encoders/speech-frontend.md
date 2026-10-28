# Speech frontend and causal convolution

Measured on 2026-10-25, Apple M2 Max, 32 GiB RAM, macOS 27.2, Go 1.25.3.
Both implementations use eight Go workers and Darwin Accelerate, without Metal
or Core ML. The reference includes the preceding Laya/Voxtral encoder and Qwen
worker changes; it differs only in the speech frontend and offline causal
convolution paths covered here.

## Changes

Voxtral and Parakeet now dispatch complete spectral frames instead of three
small matvecs per frame. A worker performs the original SIMD dot products,
power calculation and mel projection with private scratch. Model-specific
windowing, preemphasis, log floors and normalization remain unchanged. Non-F32
filterbanks retain their original matvec path, without nested worker dispatch.

The immutable 400- and 512-point DFT bases are initialized once per process.
Other sizes remain request-local, so model metadata cannot grow an unbounded
global cache. Voxtral streaming reuses worker scratch and places mel rows in
one contiguous allocation. Completed tasks clear references to audio and output
buffers. Cancellation is checked between frames and during offline log scaling.

Voxtral's offline causal convolutions also distribute independent output frames
across workers. They retain the same padding, dot products, bias and GELU order;
scratch is one input patch per worker, independent of clip duration. This path
is used by `TranscribeOffline` and the standalone offline transcription API.

## Component measurements

Four samples per implementation ran in ABBA ABBA order, each in a fresh test
process with a 500 ms benchmark target. Builds and tests ran outside timing
windows. Background load was not controlled. Medians:

| Workload | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Voxtral mel, 160 ms audio | 1.258 ms | 0.131 ms | 9.60× |
| Parakeet mel, 160 ms audio | 1.761 ms | 0.175 ms | 10.05× |
| Voxtral mel, 1 s audio | 4.425 ms | 0.552 ms | 8.02× |
| Parakeet mel, 1 s audio | 5.295 ms | 0.761 ms | 6.95× |
| Voxtral mel, 7 s audio | 25.526 ms | 3.358 ms | 7.60× |
| Parakeet mel, 7 s audio | 31.279 ms | 4.752 ms | 6.58× |
| Voxtral conv0, 700 input frames | 63.571 ms | 9.219 ms | 6.90× |
| Voxtral conv1, 700 input frames | 157.502 ms | 25.942 ms | 6.07× |
| Streaming push, 20 ms audio, tiny encoder | 0.089 ms | 0.055 ms | 1.60× |
| Streaming push, 160 ms audio, tiny encoder | 0.670 ms | 0.163 ms | 4.10× |

Mel measurements use a deterministic tone at 16 kHz, 128 mel bins, the model's
periodic/symmetric Hann window and its Slaney filterbank. They include padding,
windowing, DFT, mel projection and normalization. A warmup precedes timing, so
the new one-time table initialization is excluded. The reference still rebuilds
tables on each measured request. Convolution weights are synthetic at production
dimensions: 128/1280 input channels, 1280 output channels, kernel 3, stride 1/2.
The streaming cases use production frontend dimensions with tiny downstream
weights to isolate frontend overhead; they are not production-model throughput.

[All component samples, allocations and iteration counts](m2-max-speech-frontend-2026-10-25.csv).

## Complete Parakeet transcription

The local `nvidia/parakeet-tdt-0.6b-v3` Q8_0 GGUF transcribed a 3.6069375-second
recording: “Hello, this is a test of automatic audio transcription.” It was
generated locally with macOS Samantha at rate 175, then converted to mono
16 kHz float32 PCM. PCM SHA-256:
`12e96b41d89e6a893c07ce7b8e2402a26b81937e7fbd1fc6b27a9f237a37c57c`.

Four processes ran in ABBA order, each loading the model once and transcribing
three times. Model loading is outside timing. The first request is recorded but
excluded from the warm medians below; there are four warm samples per version.

| Warm request | Before | After |
| --- | ---: | ---: |
| Encoding | 372.54 ms | 338.82 ms |
| TDT decoding and text rendering | 15.37 ms | 15.05 ms |
| Complete transcription | 387.86 ms | 353.87 ms |
| Allocated bytes per request | 140,154,320 | 139,116,592 |

The complete request took **8.8% less time** on this clip. The two reference
process medians were 402.18 and 380.21 ms; the changed medians were 363.06 and
350.23 ms. This small sample does not establish a general speech throughput or
word-error-rate improvement. It does verify identical text, token sequences
and every encoder-output bit across all twelve requests. No complete Voxtral
transcription timing is claimed from the component results.

[All requests, timings and output digests](m2-max-speech-parakeet-2026-10-25.jsonl).

## Memory and validation

At seven seconds, Voxtral frontend allocations fell from about 1.47 MB to
0.84 MB per call, and Parakeet from 2.34 MB to 1.30 MB. Voxtral now uses four
allocations instead of 14–15 for this case; Parakeet uses 6–7 instead of 19–20.
The tradeoff is a lazily retained, process-wide table cache: 643,200 bytes for
Voxtral and 1,052,672 bytes for Parakeet, about 1.62 MiB if both are used.
Initial requests still allocate these tables. No extra model weights are copied.

Parallel convolution needs larger patch scratch: at eight workers, conv0 uses
12 KiB instead of 1.5 KiB and conv1 120 KiB instead of 15 KiB. These are allocation
and buffer measurements, not RSS or paging results.

Regression tests retain the original serial frontends and causal convolution
as references. They compare output bits for silence and varying signals,
centered/uncentered audio, preemphasis, F32/raw filterbanks, different frame and
stride counts, and one/three/eight workers. Additional tests cover concurrent
requests, cancellation, released tensor views and streaming chunk boundaries.
The portable and Metal/Core ML suites, targeted race tests, the scalar-SIMD
opt-out comparison, native `go vet` and all six cross-build targets passed.

## Reproduction

Build the reference and changed source with the same benchmark files:

```sh
CGO_ENABLED=1 go test -c -o /tmp/speech.test ./internal/engine
GOMAXPROCS=8 /tmp/speech.test -test.run '^$' -test.benchtime=500ms \
  -test.bench '^(BenchmarkSpeechFrontend|BenchmarkSpeechCausalConvolution|BenchmarkSpeechStreamingFrontend)$'
```

Alternate their binaries in ABBA ABBA order. For the complete model comparison:

```sh
say -v Samantha -r 175 -o /tmp/stt-speech.aiff \
  'Hello, this is a test of automatic audio transcription.'
ffmpeg -v error -i /tmp/stt-speech.aiff -ar 16000 -ac 1 -f f32le /tmp/stt-speech.f32
GOPHERLLM_STT_MODEL=/path/to/parakeet-tdt-0.6b-v3.q8_0.gguf \
GOPHERLLM_STT_PCM=/tmp/stt-speech.f32 \
GOPHERLLM_STT_EXPECT_TEXT='Hello, this is a test of automatic audio transcription.' \
  /tmp/speech.test -test.run '^TestSpeechTranscriptionCheckpoint$' -test.v
```

The checkpoint harness explicitly requires local paths and never downloads
models or audio. Speech synthesis may differ with OS voice revisions; retain
the same PCM file for both implementations.
