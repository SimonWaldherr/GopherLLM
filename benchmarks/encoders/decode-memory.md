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

Full Ministral configurations were read from the locally cached Q4_K_M GGUF metadata:
3B has width 3072/FFN 9216, 14B width 5120/FFN 16384. Only construction is
measured: savings are about 96/168 KiB per scratch object, not gigabytes of
model or KV storage. Constructor timings do not establish a speedup.

The Voxtral experiment uses the actual vocabulary width of 131072 with
synthetic weights and deliberately tiny transformer geometry (width 12, two
layers, FFN 16). Seven alternating pairs measured 32 steps at a median
13.854 ms before and 0.164 ms after. This isolates discarded projection work;
the tiny body makes the relative gain much larger than a real 4B decoder's.
It must not be interpreted as full-model transcription throughput. Background
activity was not controlled. The allocation difference is predominantly 32
discarded 512-KiB logits vectors; small pool/allocation noise remains.

An independent frozen decoder reference checks every generated logit, residual,
normalized state and chronological K/V row across 40 steps, including skipped
prefix outputs and sliding-window rollover. Existing offline/session fixtures
and full portable/Metal/Core ML suites pass. Targeted race tests pass.

## Reproduction

```sh
CGO_ENABLED=1 go test -c -o /tmp/decode.test ./internal/engine
GOMAXPROCS=8 /tmp/decode.test -test.run '^$' \
  -test.bench '^BenchmarkVoxtralTeacherForcedDecode$' -test.benchtime=7x -test.v
GOMAXPROCS=8 /tmp/decode.test -test.run '^$' \
  -test.bench '^(BenchmarkVoxtralTeacherForcedAllocations|BenchmarkMinistralOptionalScratch)$' \
  -test.benchtime=7x -test.benchmem -test.v
```

Set `GOPHERLLM_MINISTRAL_MEMORY_MODELS` to a platform path-list of the 3B and
14B GGUF files to read their actual dimensions; otherwise dimension-only default
geometries are used. Weight tensors are not loaded for this benchmark.
The checkpoint configuration also preserves the distinct query/value head
widths of Ministral 14B. Builds and tests run outside timing windows.

[Raw samples, allocation reports, source/binary hashes and limits](decode-memory-m2-max-2026-10-31.json).
