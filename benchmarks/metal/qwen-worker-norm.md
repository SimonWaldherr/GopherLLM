# Qwen recurrent head scheduling

Measured on 2026-10-25, Apple M2 Max, 32 GiB RAM, macOS 27.2, Go 1.25.3.
The reference is `62fc47f`. Both binaries use eight CPU workers and the same
existing Metal paths. Benchmark processes ran sequentially; this task's builds
and other checks ran outside timing windows. Background applications remained
active.

## Change

Each DeltaNet worker now completes its heads' RMSNorm and SiLU output gate
before returning. Previously the caller waited for every recurrent head, then
performed normalization and gating serially. Heads have disjoint output ranges;
normalization still precedes gating, with the same floating-point operations.
This applies to both serial decoding and the recurrence inside chunked prefill.

The task borrows the existing norm and gate slices and clears those references
after the worker barrier. It needs two additional slice headers and an epsilon
value (56 bytes including padding on arm64), with no new activation or weight
buffers and no additional worker dispatch.

## Recurrent kernel

`BenchmarkQwen35DeltaNet27B` includes convolution, normalization and the complete
48-head, 128-dimensional recurrence, but excludes the weight projections. Each
series runs four measurements per implementation with a one-second benchmark
target. Series 1 uses ABBA ABBA order, series 2 BAAB BAAB.

| Series | Reference median | Worker norm median | Time reduction |
| --- | ---: | ---: | ---: |
| 1 | 142.316 µs | 115.002 µs | 19.2% |
| 2 | 119.855 µs | 116.241 µs | 3.0% |

The first reference series varied considerably (119.982–160.816 µs); its larger
gain should not be treated as the expected improvement. The second series
confirms a smaller benefit. Both implementations report zero allocations per
operation. [All kernel measurements](m2-max-qwen-worker-kernels-2026-10-25.csv).

## Complete Qwen3.8-27B Q4_K_M model

Four fresh processes ran in ABBA order, each with one warmup and one measured
request: 39 prompt tokens, 32 generated tokens, temperature zero, repeat penalty
1.1, seed 41, F32 KV, core prefault and no prepared CPU weights. The existing
`TestConfiguredDecodePaging` harness clears the conversation cache per request.

| Order | Implementation | Decode tokens/s | Prefill | Request Mach page-ins |
| --- | --- | ---: | ---: | ---: |
| 1 | Reference | 5.167 | 17.938 s | 50,385 |
| 2 | Worker norm | 1.895 | 13.688 s | 63,221 |
| 3 | Worker norm | 6.412 | 4.412 s | 23 |
| 4 | Reference | 5.451 | 4.673 s | 23,540 |

These runs do **not** establish a whole-model throughput or paging improvement.
Page-ins cover the whole request and are VM events, not physical SSD reads.
All four requests have identical token and complete live-state digests. KV
storage is 9,437,184 bytes and shared Metal batch scratch is 6,422,528 bytes in
both implementations. [Loading, warmup and request measurements](m2-max-qwen-worker-model-2026-10-25.jsonl).

## Reproduction

Build the reference revision and the changed source into separate test binaries:

```sh
CGO_ENABLED=1 go test -tags metal -c -o /tmp/qwen-worker.test .
/tmp/qwen-worker.test -test.run '^$' -test.bench '^BenchmarkQwen35DeltaNet27B$' -test.benchtime=1s
GOPHERLLM_DECODE_MODEL=/path/to/Qwen3.8-27B-Q4_K_M.gguf \
  GOPHERLLM_DECODE_MODE=1 GOPHERLLM_DECODE_CYCLES=1 GOPHERLLM_DECODE_TOKENS=32 \
  /tmp/qwen-worker.test -test.run '^TestConfiguredDecodePaging$' -test.v
```

Alternate the two binaries in the orders above. The production-geometry test
compares every output, convolution and recurrent-state float with the original
graph using one, five and eight workers; five also exercises unequal head ranges.
Portable and Metal/Core ML suites, targeted Qwen race tests, native vet and all
six cross-build targets passed with the retained change.

Two other candidates were removed: recomputing ARM decay to avoid an intermediate
state write did not consistently improve latency; combining Ministral rotary and
KV dispatches lost median paired time in all three 64-step comparison sets.
