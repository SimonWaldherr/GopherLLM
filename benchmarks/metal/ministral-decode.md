# Ministral decode measurements

Measured on an M2 Max with 32 GiB RAM, macOS 27.2 and Go 1.25.3 on
2026-10-24. Model: `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`. The baseline
uses the vocabulary selection from `afa18b1`; both modes run in the same
loaded Runner with the same projection kernels.

## Complete chat requests

Each repetition warms one request, then alternates the original and parallel
vocabulary reductions in ABBA order for four cycles. Settings: eight threads,
Metal, four SIMD groups per projection threadgroup, F32 KV, Q8 CPU activations,
core prefault, no prepared weights, temperature zero, repeat penalty 1.1 and
seed 41. The prompt is:

> Erkläre in fünf Sätzen, wie ein Computerprogramm ausgeführt wird.

Its 27 prompt tokens are fully reused after warmup. Every request emits 64
tokens. The rates below are 64 divided by the median decode time of eight
requests per mode, excluding loading and prefill. As in `GenerationStats`,
the first token can use the logits already retained at the prompt boundary.

| Run | Original selection | Parallel selection | Throughput gain |
| --- | ---: | ---: | ---: |
| 1 | 74.86 tokens/s | 75.84 tokens/s | 1.31% |
| 2 | 74.72 tokens/s | 75.13 tokens/s | 0.55% |
| 3 | 57.14 tokens/s | 58.96 tokens/s | 3.20% |

Host load increased in the third run. Immediately after it, the ANE compiler
used about 92% of one CPU core, simulator/rendering processes were active and
14.6 GB of swap was occupied. This observation does not isolate the cause of
the slowdown; both selection modes slowed down together. All three runs are
included rather than quoting only the fastest result.

[All 48 request measurements](m2-max-ministral3b-generation-2026-10-24.csv)
include token and state digests. Generated text, token IDs, residual vectors
and resident KV values were bit-identical across both modes. The decoder still
borrows 470605824 bytes of additional mapped projections and copies zero weight
bytes, preserving the earlier 448.8 MiB saving. The new partial-winner buffer
contains 256 bytes for this vocabulary; driver allocation overhead is separate.

## Paired fixed decode inputs

After the requests, each of the 64 generated tokens is replayed at its own
position in ABBA order. All four calls overwrite the same cache row and must
select the same next token. Each sample averages two calls per mode; timings
include the complete decoder, output projection and token selection. This
limits drift between the two modes without relying on a cached final token.
These are fixed-input measurements, separate from the chat rates above.

| Run | Original median | Parallel median | Median paired saving |
| --- | ---: | ---: | ---: |
| 1 | 13.622 ms | 13.471 ms | 133 µs |
| 2 | 14.451 ms | 14.320 ms | 139 µs |
| 3 | 14.784 ms | 14.528 ms | 219 µs |

[All 192 paired measurements](m2-max-ministral3b-decode-pairs-2026-10-24.csv)
are retained. The paired saving is the median of each input's time difference,
so it need not equal the difference between the two medians.

## Reproduction

```sh
GOPHERLLM_MINISTRAL_MODEL=/path/to/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
  CGO_ENABLED=1 go test -tags metal . \
    -run '^TestMinistralConfiguredGeneration$' -count=3 -v
```

`MINISTRAL_GENERATION_JSON` records the request samples and
`MINISTRAL_DECODE_PAIRS_JSON` records the fixed-input pairs. Set
`GOPHERLLM_METAL_DENSE_PARALLEL_ARGMAX=0` outside this test to select the
previous reduction. Small vocabularies retain it automatically.

The test uses a cached short prompt. Longer contexts, cold loading, paging,
sampling and competing GPU work can change throughput. Earlier exploratory
request comparisons fluctuated under background load, so the measured gain
should not be read as a guarantee for every run. This change does not alter
Qwen3.8's hybrid decoder, and these measurements establish no M4 or
Ministral-14B throughput figure.
