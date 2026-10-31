# Decoder scratch and teacher-forced positions

Measured on M2 Max, 32 GiB, Go 1.27.3, 2026-10-31. These are component
measurements; no new whole-model tokens/s, peak RSS or paging result is claimed.

## Changes

Voxtral's CPU fallback now skips the vocabulary projection for teacher-forced
positions in both offline and streaming transcription. The public decoder API
still returns independent, complete logits. The CPU FFN overwrites its consumed
gate vector instead of retaining a third activation vector, saving one
`4 * IntermediateSize` byte slab per decoder state. The Metal path already
skips unused projections and is unchanged.

Generic decoder scratch no longer eagerly allocates fused gate/up and MTP
inputs. Their consumers allocate the required sizes on first use. Ministral
does not use either buffer in ordinary decoding. Explicit sizing before the
fused projection preserves models whose weights infer dimensions from output
storage, including Falcon.

## Measurements and limits

| Component | Previous allocation | Changed allocation |
| --- | ---: | ---: |
| Ministral 3B scratch construction | 907,837 B | 809,501 B |
| Ministral 14B scratch construction | 1,126,626 B | 954,605 B |
| Voxtral, 32 teacher-forced steps, warmed state | 16,784,115 B | 768 B |

The historical Ministral measurements used configurations read from locally
cached Q4_K_M GGUF metadata: 3B has width 3072/FFN 9216, 14B width 5120/FFN
16384. This omitted the tensor-shape inference that `LoadModel` applies after
reading metadata. In particular, 14B's metadata-derived head width was 160,
whereas its attention tensors resolve to 128. Its absolute allocation totals
above therefore describe that metadata-only geometry, not the loaded decoder's
final geometry. The optional GateUp/MTP slabs depend on model/FFN width, so the
96/168-KiB savings per scratch object remain unchanged. Only construction is
measured; these are not gigabytes of model/KV savings or a decoding speedup.

The Voxtral experiment uses the actual vocabulary width of 131072 with
synthetic weights and deliberately tiny transformer geometry (width 12, two
layers, FFN 16). Seven alternating pairs measured 32 steps at a median
13.854 ms before and 0.164 ms after. This isolates discarded projection work;
the tiny body makes the relative gain much larger than a real 4B decoder's.
It must not be interpreted as full-model transcription throughput. Background
activity was not controlled. The allocation difference is predominantly 32
discarded 512-KiB logits vectors; small pool/allocation noise remains.

An independent frozen decoder reference checks every generated logit, residual,
normalized state and chronological K/V row across 40 steps. Tests cover windows
of 1, 3 and 16 positions with skipped prefix outputs, alternating output requests
and outputs at every step. Existing offline/session fixtures and full portable,
Metal/Core ML and targeted race suites pass.

## Reproduction

```sh
CGO_ENABLED=1 go test -c -o /tmp/decode.test ./internal/engine
GOMAXPROCS=8 /tmp/decode.test -test.run '^$' \
  -test.bench '^BenchmarkVoxtralTeacherForcedDecode$' \
  -test.benchtime=1s -test.count=10 -test.benchmem
GOMAXPROCS=8 /tmp/decode.test -test.run '^$' \
  -test.bench '^BenchmarkMinistralOptionalScratch$' \
  -test.benchtime=1s -test.count=10 -test.benchmem
```

The current Voxtral harness reports separate `reference=true/false` subbenchmarks
with standard Go timing/allocation results and a `steps/s` metric. One operation
is 32 teacher-forced steps. Each variant fills its sliding window before timing;
setup, warmup and logging are excluded. This replaces the combined paired-log
timing benchmark and the duplicate allocation benchmark. The linked historical
JSON retains its original samples and method. New harness output should not be
mixed with those samples. Repeat runs on an idle host and inspect the spread;
running the two subbenchmarks in separate processes with alternating order can
help identify host drift.

Set `GOPHERLLM_MINISTRAL_MEMORY_MODELS` to a platform path-list of the 3B and
14B GGUF files. The current harness applies the same tensor-shape inference as
`LoadModel`, including 14B's resolved 128-wide query/value heads; its default
geometry uses that width too. Tensor payloads are not loaded for the scratch
benchmark. The historical JSON retains the earlier metadata-only configuration
and samples, so do not mix its absolute allocation totals with current output.
Builds and tests run outside timing windows.

The [current component harness](../../internal/engine/decode_memory_optimization_test.go)
also contains the parity tests. They can run without Metal or model weights:

```sh
CGO_ENABLED=0 go test ./internal/engine \
  -run '^TestVoxtralDecodeProjectionAndScratchParity$' -count=1
```

[Raw samples, allocation reports, source/binary hashes and limits](decode-memory-m2-max-2026-10-31.json).

## Reusing decoder inputs (2026-10-31)

Offline and streaming Voxtral now reuse an owned token/audio input vector. Each
lookup overwrites the entire vector before adding audio, leaving model weights,
audio inputs and publicly returned embedding rows untouched. CPU and Metal
consume this input synchronously. Closing a session releases its scratch.

Teacher-forced CPU steps also avoid the heap-allocated logits slice header:
only output positions call the allocating projection. The public decoder still
returns independent logits. Seven alternating before/after process pairs gave
**768 to 0 B per 32 warmed steps**, with medians of 163.425 and 161.948 µs.
That small timing difference does not establish a decoding speedup.

The separate embedding/fusion benchmark uses synthetic F32 weights, vocabulary
32 and a hot embedding row. At width 3072, seven samples per variant measured
**24,576 to 0 B per position**, with median time **11.206 to 2.005 µs**. This is
the former two-slab offline input preparation versus reused storage, without
decoder work. Streaming previously allocated one embedding slab per position;
it now reuses that slab too. Neither benchmark measures full transcription,
real-model RSS, cold embedding lookups or paging.

```sh
CGO_ENABLED=1 go test ./internal/engine -run '^$' \
  -bench '^BenchmarkVoxtralFusedEmbedding$' -benchtime=200ms -count=7 -benchmem
CGO_ENABLED=1 go test ./internal/engine \
  -run '^(TestVoxtralSessionEmbeddingReuseMatchesReference|TestVoxtralTeacherForcedDecodeReusesScratch)$' \
  -count=1
```

A 64-step independent-reference test checks generated tokens and K/V state
through sliding-window rollover, plus input/weight ownership and scratch release.
Malformed time dimensions, non-finite frequencies and mismatched FFN widths
return errors. Whole-recording transcription rejects invalid PCM before warmup.

[Raw component samples and source/binary hashes](voxtral-input-reuse-m2-max-2026-10-31.json).
