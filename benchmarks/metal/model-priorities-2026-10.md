# Model priorities and measured changes — October 2026

This is an engineering priority for GopherLLM on Macs with 32 GiB RAM, not a
prediction of market share or an independent model-quality ranking. Sources
were checked on 2026-10-24. The measured paths below generate text.

## Priority

1. **Qwen:** small dense/hybrid models and Qwen3.8-27B for coding and general
   local use; Qwen3.6-35B-A3B for sparse inference. The latter uses 35B total
   parameters, 3B active and hybrid DeltaNet/attention layers. Optimizing the
   shared architecture benefits multiple checkpoints. Sources:
   [27B model card](https://huggingface.co/Qwen/Qwen3.8-27B),
   [35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).
2. **Gemma 4:** E2B/E4B for smaller devices, 12B for a larger dense option,
   and 26B-A4B for desktop MoE. E2B's 2.3B effective parameters coexist with
   5.1B total parameters because of large per-layer embedding tables. Those
   tables need row lookups, which makes their storage policy especially useful
   to optimize. Sources: [E2B](https://huggingface.co/google/gemma-4-E2B-it),
   [26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it).
3. **Ministral 3:** retain 3B as the low-latency baseline, with 8B/14B as larger
   options. The family targets edge deployment and includes multilingual and
   tool-use capabilities. Source:
   [Ministral 3 model card](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512).

Next candidates are [Nemotron 3 Nano's hybrid MoE architecture](https://research.nvidia.com/labs/nemotron/Nemotron-3/)
and European models such as [OpenEuroLLM Prelude](https://openeurollm.eu/prelude).
Their relevance warrants evaluation, not an unmeasured performance claim.
[Aleph Alpha Kolibri 1](https://huggingface.co/Aleph-Alpha/Kolibri-1) is interesting
for German/English, but its 78B total parameters make it a lower priority for
this RAM budget. A four-bit payload alone would be roughly 39 GB before scales,
KV and runtime overhead. MoE active parameter counts do not describe total RAM.

## Retained changes

- Native Gemma no longer prepares accelerator/precomputed-scale resources for
  its per-layer token lookup table or an untied input embedding. Borrowed scalar
  lookup tables stay packed; tied vocabulary heads retain their projection path.
- `--prefault core` skips Gemma's per-layer token lookup table. The local E2B
  Q4_K_M contains **1926758400 bytes (1.79 GiB)** of Q6_K PLE rows. They are
  demand-paged instead of all being touched at startup. This is avoided eager
  file access, not a measured 1.79 GiB reduction in steady-state RSS or swap.
- Qwen Q8_0 SwiGLU and Gemma GeGLU reuse the consumed gate buffer for activation.
  Qwen Q4_K already did this. Generic Q4_K prefill retains its separate hidden
  buffer: the candidate regressed the measured Ministral 32-position shape.
- Native Gemma uses parallel 4096-entry vocabulary reductions and a merge for
  eligible greedy selection. The existing penalty order, finite checks and
  lowest-index tie rule are preserved. Small vocabularies retain the original
  reduction; nonlinear softcap plus repetition penalty still uses the existing
  materialized-logit fallback.

## Measurements

Apple M2 Max, 32 GiB RAM, macOS 27.2, Go 1.25.3. Background application and GPU
load varied. Each request recomputes its prompt; warmups/loading are separate.
Eight CPU threads, Metal, F32 KV, Q8 CPU activations, seed 41, temperature zero,
no prepared CPU weights and core prefault. Prompt:

> Erkläre in fünf Sätzen, wie ein Computerprogramm ausgeführt wird.

For eligible GPU greedy selection, repetition penalty is **1**. E2B uses
38 prompt / 32 generated tokens and two ABBA cycles per run. A4B uses
41 prompt / 16 generated tokens and one cycle. Both modes use the new lookup
loading policy; the alternating comparison measures runtime switches only.

| Complete request | Reference tokens/s | Optimized tokens/s |
| --- | ---: | ---: |
| E2B, run 1 | 24.67 | 26.77 |
| E2B, run 2 | 18.75 | 15.94 |
| E2B, run 3 | 26.66 | 29.90 |
| 26B-A4B, one run | 8.94 | 9.38 |

These rates divide the generation cap by median decode time. The inconsistent
E2B direction and small A4B sample do **not** establish a stable whole-model
speedup. An earlier E2B run at penalty 1.1 measured 21.99 / 16.03 tokens/s;
that configuration cannot use the GPU greedy shortcut and is also retained.

Fixed-input ABBA pairs isolate the changed output projection/selection path
with the same real model weights and normalized vector:

| Model | Reference median | Parallel median | Median paired saving | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| E2B | 1.766 ms | 1.469 ms | 278 µs | 85 / 96 |
| 26B-A4B | 2.664 ms | 2.397 ms | 282 µs | 30 / 32 |

This includes the vocabulary projection and GPU winner readback, not the whole
decoder. Parallel selection adds 512 bytes of shared scratch at 262144 entries.
All request text, tokens, final hidden/normalized vectors and resident KV values
were bit-identical. The output pairs also require identical winning tokens.

Fresh-process FFN workspace measurements use synthetic quantized weights at
representative shapes and 32 positions. These are exact retained buffer lengths,
excluding model weights, per-weight vectors, other workspaces and driver memory:

| Shape / format | Reference | Retained implementation | Saving |
| --- | ---: | ---: | ---: |
| Ministral 3072 → 9216 → 3072, Q4_K/Q6_K | 4.125 MiB | 4.125 MiB | unchanged |
| Qwen 2560 → 9216 → 2560, Q8_0 | 4.000 MiB | 2.875 MiB | 1.125 MiB |
| Gemma 1536 → 6144 → 1536, Q4_K/Q6_K GeGLU | 2.625 MiB | 1.875 MiB | 0.750 MiB |

The isolated retained Qwen/Gemma kernels were approximately neutral in latency;
this change is a memory saving. The removed Ministral candidate lost a median
175 µs per 32-position FFN pair and is marked `retained=False` in the pair data.
A fresh E2B process also confirms its larger layers' whole shared-workspace
saving: **5964040 → 4391688 bytes**, about **1.50 MiB** net, including the
new 512-byte selection scratch. The reference is the first baseline warmup in
the greedy runs, before candidate buffers were allocated. The final fresh-process
check used the same prompt batch with an eight-token output cap. Its timings
are excluded because tests/builds ran concurrently; neither output length grows
these shared buffers beyond the prompt batch allocation.

Raw data:

- [E2B greedy requests and output pairs](m2-max-gemma-e2b-greedy-2026-10-24.jsonl)
- [E2B with penalty 1.1](m2-max-gemma-e2b-default-2026-10-24.jsonl)
- [26B-A4B requests and output pairs](m2-max-gemma-a4b-greedy-2026-10-24.jsonl)
- [Fresh E2B memory check; timing fields omitted](m2-max-gemma-e2b-memory-2026-10-24.jsonl)
- [FFN workspace bytes and output digests](m2-max-shared-ffn-memory-2026-10-24.csv)
- [FFN pairs, including the reverted candidate](m2-max-shared-ffn-pairs-2026-10-24.csv)

## Reproduction and validation

```sh
CGO_ENABLED=1 go test -tags metal -c -o /tmp/gopherllm-models.test ./internal/engine
GOPHERLLM_DECODE_MODEL=/path/to/gemma-4-E2B-it-Q4_K_M.gguf \
  GOPHERLLM_DECODE_TOKENS=32 GOPHERLLM_DECODE_PENALTY=1 \
  /tmp/gopherllm-models.test -test.run '^TestConfiguredDecodePaging$' -test.count=3 -test.v
# Run each combination in a fresh process; buffers retain their high-water mark.
GOPHERLLM_TEST_FFN_KIND=qwen GOPHERLLM_METAL_FFN_INPLACE=1 \
  /tmp/gopherllm-models.test -test.run '^TestMetalFFNWorkspace$' -test.v
```

For the FFN test, use `qwen`, `gemma` or `ministral` and modes `0` / `1`.
Add `GOPHERLLM_TEST_FFN_PAIRS=1` for timed component pairs. For a fresh-process
whole-model memory sample use `GOPHERLLM_DECODE_MODE=1` and
`GOPHERLLM_DECODE_CYCLES=1`. Avoid simultaneous builds for timing runs.

Runtime opt-outs: `GOPHERLLM_METAL_FFN_INPLACE=0` and
`GOPHERLLM_METAL_GEMMA_PARALLEL_ARGMAX=0`. The previous Qwen-only activation
opt-out remains supported. `--prefault all` explicitly warms the entire mapping.

Validation covers scalar lookup ownership and exact logits with tied/untied
heads, F32/F16/BF16, prefault ranges, mixed quantized FFNs and batch boundaries,
argmax ties/repeated penalties/non-finite values, and concurrent users of shared
scratch. Portable/Metal test suites, native race tests, vet and cross-builds
were run after the final change.
