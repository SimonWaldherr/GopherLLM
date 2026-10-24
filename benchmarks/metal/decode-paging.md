# Decoding and memory on a 32 GiB Mac

Measured on an Apple M2 Max with 32 GiB RAM, macOS 27.2 and Go 1.25.3,
on 2026-10-24. The reference uses the decoding paths from `4caf893`.
Both modes run in the same binary; the switches below select the old paths.
Background simulators, rendering and other applications remained active.
These results do not establish M4 throughput or performance on an idle Mac.

## Changes

- Dense attention with at most 128 visible positions computes its 32-position
  tiles and their ordered merge in one threadgroup. This saves one GPU dispatch
  per layer without changing the softmax reduction or KV representation. Longer
  contexts retain the tiled implementation.
- Qwen's direct Q6_K QKV projection now overlaps the independent CPU gate and
  alpha/beta projections. GPU completion still precedes the recurrent update.
  Input/output buffers and mapped weights are reused. There is no additional
  worker goroutine, expanded weight copy or recurrent-state copy.
- Qwen's SiLU activation overwrites the consumed gate slab. This removes one
  process-wide hidden slab, rather than a buffer per layer. Other model paths
  retain their existing workspace.

## Method

`TestConfiguredDecodePaging` uses eight CPU threads, Metal, four SIMD groups
per dense projection threadgroup, F32 KV, Q8 CPU activations, core prefault,
no prepared CPU weights, temperature zero, repeat penalty 1.1 and seed 41.
The prompt is:

> Erkläre in fünf Sätzen, wie ein Computerprogramm ausgeführt wird.

Ministral renders 27 prompt tokens and generates 64; Qwen renders 39 and
produces 32. Every request clears the conversation cache, so prefill is recomputed.
Loading and two pipeline warmups are recorded separately. Two ABBA cycles
produce four requests per mode per run. Throughput is the output token count
divided by median decode time, using `GenerationStats` (the first output token
uses the final prompt logits). This is not total request throughput.

Text, token IDs, final residual/normalized vectors, resident KV values and
Qwen's complete convolution/recurrent state must be bit-identical between modes.
The separate dense fixture also retains the unmodified binary's state hash:
`e2f9a6b893c698f384f3c27205ef5ad3aad9a3296f5f1579cc9f5f5e1403a9cc`.

## Ministral 3B Q4_K_M

Model: `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`, 2146498240 bytes.

| Run | Reference tokens/s | Short attention tokens/s | Gain |
| --- | ---: | ---: | ---: |
| 1 | 75.20 | 75.38 | 0.24% |
| 2 | 74.73 | 75.33 | 0.80% |
| 3 | 74.79 | 75.08 | 0.39% |

Across the 24 requests, the two medians are 74.84 and 75.13 tokens/s (+0.40%).
Median prefill is 156.04 and 156.48 ms; no prefill gain is established.
After each run, all 64 decode inputs are replayed at identical positions in
ABBA order. The median paired saving is 54.5, 58.8 and 56.4 microseconds per
complete step. The gain is small, but has the same sign in all three runs.

The dense decoder still borrows 470605824 bytes of mapped projections and
copies zero weight bytes. Allocated KV remains 28327936 bytes, prefill scratch
11854200 bytes and shared Metal batch scratch 540672 bytes. Sampled requests
had no Mach page-ins. This does not measure performance under Ministral paging.

## Qwen3.8-27B Q4_K_M

Model: `Qwen3.8-27B-Q4_K_M.gguf`, 16810714336 bytes, `qwen35` architecture.

| Comparison | Reference tokens/s | Optimized tokens/s | Reference / optimized median prefill |
| --- | ---: | ---: | ---: |
| Overlap only | 6.34 | 6.73 | 5.83 / 5.88 s |
| Overlap + reused gate slab | 3.64 | 6.43 | 10.78 / 6.64 s |

The first run observed a 6.17% throughput increase. The second run encountered
302879 reference versus 16956 optimized Mach page-ins across its sampled
requests; its large apparent gain cannot be attributed to the code change.
Conversely, the first run had 7302 versus 155922 page-ins, with one slow
optimized request. All samples, including these outliers, are retained.
There is **no stable whole-model speedup estimate** under this host load.

A subsequent fresh optimized process measured 6.09 tokens/s after warmup.
Eight fixed-input whole-forward ABBA pairs then restored the entire recurrent
state before each call. They measured medians of 225.95 / 213.52 ms, but only
three of eight individual pairs improved; the median paired saving was
**-6.07 ms**. This check also fails to establish a consistent whole-model gain.
It includes full vocabulary logits, unlike the greedy chat selection above.
Checkpoint restoration and state hashing occur outside the timed interval;
the test-only checkpoint adds about 150 MiB after the memory samples.

The isolated schedule change is clearer: 32 ABBA pairs of the first DeltaNet
layer's real Q6_K QKV and CPU gate/alpha/beta projections measured median
latencies of **1.258 → 0.986 ms**. The median paired saving was **247 µs**,
with 31 of 32 pairs faster. These use one fixed final normalized input and
verify every projected float; they are component latencies, not tokens/s.
The full-model and component evidence should not be conflated.

Raw data:

- [Ministral: all three runs and 192 decode pairs](m2-max-ministral3b-attention-2026-10-24.jsonl)
- [Qwen: overlap-only run](m2-max-qwen27b-overlap-2026-10-24.jsonl)
- [Qwen: final alternating run](m2-max-qwen27b-decode-2026-10-24.jsonl)
- [Qwen: fresh-process memory and fixed-input pairs](m2-max-qwen27b-memory-pairs-2026-10-24.jsonl)

## Memory and paging interpretation

The removed slab contains `32 * 17408 * 4 = 2228224` bytes (2.125 MiB) for
Qwen's maximum 32-position verification/prefill block. This is activation
scratch, not a reduction in the 16.81 GB GGUF or its mapped working set.
An alternating run retains the reference slab after it has been allocated;
separate processes are necessary to measure the normal retained-buffer saving.

The fresh reference process allocated **8650752 bytes (8.25 MiB)** of shared
Metal batch scratch on its first warmup; the fresh optimized process allocated
**6422528 bytes (6.125 MiB)** and retained that size through its measured
request. The difference is exactly 2228224 bytes, or 25.8% of this workspace.
Allocated KV remains 9437184 bytes in both modes. The optimized process still
had 694503 page-ins during its cold warmup and 9694 in its measured request:
this optimization does not prevent paging under competing application load.

Mach resident size, charged physical footprint, peak RSS and Go heap are
recorded separately. They account for shared/file-backed GPU pages differently
and must not be added together or treated as the model's total RAM requirement.
Page-ins are VM events, not a count of physical SSD reads. `getrusage` minor
fault deltas can be negative on this macOS version; the Mach counters avoid
using that derived quantity to infer paging improvements.

The optimizations preserve existing mapped-weight savings. A 2.125 MiB scratch
saving alone cannot eliminate system-wide swap pressure. The data establish no
reduction in SSD traffic or guaranteed swap-free operation at long contexts.

## Reproduction

Build once, then run model measurements sequentially without simultaneous builds:

```sh
CGO_ENABLED=1 go test -tags metal -c -o /tmp/gopherllm-decode.test .
GOPHERLLM_DECODE_MODEL=/path/to/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
  /tmp/gopherllm-decode.test -test.run '^TestConfiguredDecodePaging$' -test.count=3 -test.v
GOPHERLLM_DECODE_MODEL=/path/to/Qwen3.8-27B-Q4_K_M.gguf GOPHERLLM_DECODE_TOKENS=32 \
  /tmp/gopherllm-decode.test -test.run '^TestConfiguredDecodePaging$' -test.v
```

For separate-process memory samples, add `GOPHERLLM_DECODE_MODE=0` or `1`
and `GOPHERLLM_DECODE_CYCLES=1`, launching each in a fresh process. Add
`GOPHERLLM_DECODE_QWEN_PAIRS=1` for the optional recurrent-checkpoint forward
comparison after the memory measurements. Qwen projection pairs run automatically. The test
sets its own baseline/candidate switches and emits `DECODE_PAGING_JSON` rows.
Outside the test, each optimization is enabled by default and can be disabled
independently:

- `GOPHERLLM_METAL_DENSE_FUSED_ATTENTION=0`
- `GOPHERLLM_QWEN_OVERLAP_INPUTS=0`
- `GOPHERLLM_METAL_QWEN_INPLACE_SILU=0`

Residual/norm fusion and a combined Qwen gate/up/SiLU projection were also
measured during development. Neither was retained: norm fusion gave no stable
Ministral gain, and combined Qwen projections regressed the isolated kernels.

## Validation

Native Metal and portable Go suites, `go vet -tags metal ./...`, the targeted
Go race tests and `make cross-build` cover the changes. Native fixtures check
attention boundaries through 130 positions, sliding windows, serial/concurrent
Metal encoders, both FFN down-projection quantizations, vector-kernel fallback,
invalid inputs and completion after a CPU callback panic. Real-model comparisons
also check the complete live state, not only generated text.
