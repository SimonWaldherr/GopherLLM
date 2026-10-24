# Performance and memory use

Start with the [runtime recommendations](../README.md#performance-notes) and
[benchmark commands](../README.md#benchmarking-and-profiling). This reference
records implementation constraints and measurements from different stages of
development. Numbers below describe their stated model, backend and workload;
they are not a current cross-model leaderboard or additive speedups.

- [Prompt preparation](#prompt-preparation)
- [CPU bandwidth measurements](#cpu-bandwidth-measurements)
- [Runtime choices](#runtime-choices)
- [Metal](#metal)
- [CPU kernels](#cpu-kernels)
- [KV cache and paging](#kv-cache-and-paging)
- [M2 Max, M4 and M4 Pro Macs](#m2-max-m4-and-m4-pro-macs)
- [Environment variables](../README.md#environment-variables)

For complete recent experiments with raw data and reproduction commands, see
[Ministral Metal decoding](../benchmarks/metal/ministral-decode.md),
[Core ML embeddings](../benchmarks/coreml/README.md), and
[Core ML FFNs](../benchmarks/coreml/ffn.md).

## Prompt preparation

- Qwen ChatML and hybrid renderers tokenize fixed separators once per render,
  preserving the existing tokenization boundaries. On the M2 Max, preparing a
  hybrid prompt with 32 tool results fell from 358 to 238 microseconds with 44%
  fewer allocations. A Qwen3 history with 65 input messages used 20% fewer
  allocations with approximately unchanged render time. These are prompt-preparation
  measurements, not transformer-prefill speedups.
- Bounded streaming requests reuse their first context preflight, including
  conversation recall. The request owns the rendered token IDs; every message
  and option must match before the single-use plan is consumed. Later tool
  rounds plan again. Library integrations can use `PrepareChatGeneration` and
  `PreparedChat.Wrap` with their existing generator. On the M2 Max, a synthetic
  128-message server benchmark reduced preflight time by 48% without recall
  and 40% with recall; these figures exclude transformer computation.

## CPU bandwidth measurements

- **Weight streaming can limit CPU decode.** Measured on an i7-10850H laptop with dual-channel DDR4-2933 with Ministral-3 3B Q4_K_M, which streams ~2.2 GB of weights per
  generated token:

  | Measurement | Throughput |
  | --- | --- |
  | `MatvecQ6KInto`, 330 MB DRAM-resident, 12 threads | ~22-25 GB/s |
  | Pure read of the same footprint, no weight decode | ~28-33 GB/s |
  | Same int8 row kernel on L2-resident weights, 12 threads | ~52-58 GB/s |

  These measurements suggest a bandwidth bottleneck for this CPU workload.
  Reducing bytes read or sharing weight reads across tokens can help more than
  faster arithmetic alone. They do not establish a roofline for GPU decoding,
  long-context attention, or other machines.
  Thread counts of 6, 8 and 12 measured within noise here; 4 was slower.
  Calibrate thread count on the target system instead of generalizing this result.
- **The older CPU batching path amortized about 1.7× on this DDR4 laptop.**
  With a DRAM-resident weight, that version of `matvecBatchQ8`'s cost per token
  fell from ~18 ms at p=1 to ~11 ms and then flattened. Each extra position still
  cost ~0.6–0.7 of a full pass because the int8 kernel re-decoded the weight row
  per token. A three-position verification batch cost about 2.4 passes, limiting
  the potential gain to roughly 1.2× even with perfect draft acceptance.
  (An n-gram/prompt-lookup drafter
  was built and measured: it produced bit-identical output but ran at 0.81x,
  since real acceptance was ~39% on a 17% drafter hit rate.) These measurements
  describe that CPU kernel and experiment. The current multi-token SIMD kernels,
  short GPU verifier and adaptive drafting need their own end-to-end measurements
  on each target machine; see [Speculative decoding](../README.md#speculative-decoding).

## Runtime choices

- Use `--threads <N>` to set both GopherLLM worker threads and `GOMAXPROCS`.
  Make targets expose the same setting as `THREADS=<N>`; 8 was fastest in the
  measured M2 Max setup, but should be re-benchmarked on each target Mac.
- The short-context attention path avoids constructing an escaping per-layer
  worker closure. On the M2 Max development machine this reduced a tiny-model
  forward pass from 2,502 to 2,339 ns/op, allocations from 11 to 8, and a real
  0.11B F32 GGUF generation benchmark from 39.150 to 38.081 ms/op.
- Use `--prepare-quant` when slower startup is acceptable; it precomputes Q4_K
  scale/min data plus selected Q6_K scale data, then switches supported rows to
  prepared kernels. Untied packed input embeddings use row lookups and need
  neither prepared CPU scale/min arrays nor Metal handles. For Ministral 3 14B
  Q4_K_M this avoids 160 MiB of prepared tables; `--prefault core` also leaves
  its 360 MiB packed input embedding demand-paged. These are tensor/array sizes,
  not measurements of total RSS or paging performance. Tied embeddings, including
  Ministral 3 3B's vocabulary output, retain their output-projection resources.
- Use `--out-of-core` when a GGUF, especially a large sparse-MoE
  model, does not fit comfortably in RAM or Apple unified memory. It keeps the
  model CPU-only, disables Metal and prepared-quant copies, leaves F16/F32/BF16
  matrices as mmap-backed scalar bytes, and does not prewarm the rank-3 expert
  banks (including fused `ffn_gate_up_exps` layouts). The operating system
  pages selected experts in on demand; this is not
  a hard RSS limit, so a cold expert can add SSD/page-fault latency and dense
  models that are far larger than RAM can still thrash.
  Split GGUFs named `-00001-of-000NN.gguf` are supported. Each shard stays
  mapped independently and no
  merged copy is materialised, so peak memory tracks resident pages rather than
  model size; the default (non-out-of-core) split path still concatenates the
  shards into one buffer, which needs RAM for the whole model.
  It intentionally rejects `--metal`, `--prepare-quant`, `--auto`, and byte-backed loads.
  Library users can use
  `gopherllm.WithOutOfCore(true)` and optionally
  `WithMmapPrefault(gopherllm.MmapPrefaultNone)` for fully lazy paging.
- Use `--temp 0 --top-k 1` for deterministic greedy output.
- Use `--min-p <F>` (e.g. `0.05`) for min-p nucleus sampling; `0` disables it.
- `--bench-json` and `--kernel-bench-json` are intended for repeatable performance
  comparisons.

## Metal

- Metal requires a build with `CGO_ENABLED=1 -tags metal` (the plain `make
  build`/`run` targets do this automatically on macOS when Xcode
  Command Line Tools are present; `make build-metal` does it explicitly
  regardless of platform detection) and must be enabled with `--metal` at
  runtime (also automatic from the `make` targets; pass it yourself for
  a manually built binary). Supported dense Ministral/Mistral 3 and Qwen 3
  models now use a complete Metal decoder by default: normalization, attention,
  Q4_K/Q6_K/Q8_0 projections, FFN and vocabulary output share one command buffer.
  Independent projections run concurrently, and aligned F32 KV caches share
  pinned memory with Go instead of copying the prefix each token. This path
  requires 128-dimensional heads and at most 16384 allocated KV slots, and
  covers Ministral 8B's interleaved sliding-window/full-attention layer
  pattern (each layer's window is bound individually, so a uniform window on
  every layer and this interleaved pattern are equally supported). Unsupported
  configurations retain the selective path. Set `GOPHERLLM_METAL_DENSE_DECODE=0`
  to compare with that fallback.
  The selective
  path fuses sufficiently large mixed Q4_K/Q4_K/Q6_K Q/K/V projections into
  one command buffer and offloads large Q4_K projections, Q4_K gate/up + SiLU
  + Q6_K FFN-down in one command buffer, and Q6_K vocabulary-output
  projections. Narrow GQA K/V and 3K-5K-row attention-output projections stay
  on the faster CPU Q8_K/SIMD path; on the measured M2 Max, forcing those
  Ministral 3B/14B shapes through Metal was 35-65% slower per kernel.
  Deterministic decode
  (`--temp 0` or `--top-k 1`) also applies the bounded recent-token repeat
  penalty on-device before reducing the vocabulary output to its argmax. This
  keeps the default repeat penalty from forcing a 131k-logit readback and CPU
  scan on Ministral 3B. GGUF files opened
  through mmap are exposed to Metal as shared no-copy weight buffers;
  the complete decoder also uses no-copy views for projections registered on
  demand, after verifying that their raw bytes belong to the Runner's OS
  mapping. Go-owned matrices, including loader-padded weights, and byte-backed
  models retain owned copies for cgo safety. Prepared ARM64 kernels remain as
  the fallback for small projections and Metal failures. Local Ministral 3
  Q4_K_M checks eliminated 448.8 MiB of duplicate GPU weight buffers for 3B
  and 1150.8 MiB for 14B; mapped weight pages remain under OS control. The
  path remains experimental; use
  `--kernel-bench-json` and `--bench-json` on the target Mac before deployment.
- Metal also supports large Q8_0 FFNs and vocabulary projections, including
  on-device greedy argmax with repeat penalties. This enables selective GPU
  offload for Qwen Q8_0 while its narrow attention projections stay on CPU.
  Q4_K/Q6_K prefill uses four-token kernels with adjacent SIMD lanes reading
  adjacent quantized values; the selective single-token fallback retains its previous kernels.
  Metal uses float activations and can produce different text from the CPU
  int8-activation path.
- Dense Gemma uses fused quantized Gate/Up → tanh-GELU → Down on Metal,
  including the narrower FFNs in Gemma 4 E2B. Native Gemma 4 supports batched
  Metal FFNs during prefill; attention, shared KV and per-layer embeddings keep
  their existing semantics. Q4_K/Q6_K/Q8_0 vocabulary projections use vectorized
  Metal kernels. Set `GOPHERLLM_METAL_GEMMA=0` to compare the previous path.
  GeGLU and Qwen Q8_0 SwiGLU reuse the consumed gate buffer for activation;
  `GOPHERLLM_METAL_FFN_INPLACE=0` restores a separate activation buffer.
  Gemma greedy selection reduces large vocabularies in parallel. On M2 Max,
  real E2B/26B-A4B output projection/selection pairs saved about 0.28 ms;
  whole-model results varied with host load. Repetition penalties combined
  with nonlinear logit softcapping retain the materialized-logit path.
  Set `GOPHERLLM_METAL_GEMMA_PARALLEL_ARGMAX=0` for comparison. See the
  [model priorities and measured changes](../benchmarks/metal/model-priorities-2026-10.md)
  for memory savings, raw measurements and the rejected Q4 prefill candidate.
- Prefill batches of at least 16 tokens use 32×32 tiled Q4_K/Q6_K/Q8_0
  matrix kernels with Metal SIMD-group matrix multiply-accumulate and float32
  operands. Weights are decoded into shared tiles without expanding the model.
  Sixteen-wide reduction tiles reuse their storage for the output tile, cutting
  threadgroup memory from 8 to 4 KiB.
  Smaller batches and unavailable pipelines retain the previous kernels.
  Contracting Q4_K down projections now also participate in the complete
  GPU-resident FFN. This covers the half of Ministral Q4_K_M layers that
  previously missed FFN fusion because their down projection was not Q6_K.
- Eligible dense Ministral/Qwen/Llama chunks keep normalization, RoPE,
  attention, residuals and FFN intermediates on the GPU in one command buffer.
  Prefill and verification fuse each residual addition with the following
  RMS normalization, preserving the existing float32 state and reduction order.
  New KV rows are published only after successful completion and a cancellation
  check. Retained scratch, including pending KV, is capped at 256 MiB; attention
  partials are capped at 64 MiB. Unsupported shapes and smaller chunks retain
  the existing path. Set `GOPHERLLM_METAL_WHOLE_PREFILL=0` to compare it.
  With real Ministral 3 3B Q4_K_M weights on the M2 Max (eight threads), warmed
  transformer-body prefill measured 186 → 156 ms for 32 tokens and 454 → 404 ms
  for 128 tokens, with no steady-state allocations. These measurements exclude
  model loading and the vocabulary projection.
  With mapped-weight reuse and residual/normalization fusion, complete CLI
  requests measured 265 → 200 ms for Ministral 3 3B and 723 → 567 ms for 14B
  uncached prefill (27 prompt tokens, Q4_K_M, Metal, eight threads on a 32-GB
  M2 Max). `scripts/benchmark_mistral.py` checked identical greedy text over
  three alternating before/after pairs, generating 64/32 tokens respectively.
  Cached decode medians were 978 → 1009 ms (3B) and 1435 → 1471 ms (14B).
  A follow-up with six cached samples per version measured 929 → 931 ms (3B) and
  1582 → 1589 ms (14B), so the initial slowdown was not stable. Background-load
  variation was present; these runs establish lower prefill time and fewer
  weight copies, not higher decoding throughput.
- Dense Metal decode reads scheduling overrides once per token, skips shared-KV
  transfer calls and omits the second Q/K barrier when neither head is normalized.
  Auto mode measures two, four and eight SIMD groups per threadgroup for the
  loaded dense graph, retaining four until calibration shows a sufficient gain.
  Each row keeps its previous lane assignment and floating-point reduction order.
  The mapped-weight savings remain 448.8 MiB (3B) and 1150.8 MiB (14B), with no
  copied weights in the dense decoder. On the same M2 Max, host command encoding
  measured 0.313 → 0.258 ms/token (3B) and 0.497 → 0.388 ms/token (14B).
  Sixteen alternating samples per version in one loaded Runner measured cached
  decode medians of 1.062 → 1.057 s for 64 generated tokens (3B) and
  1.800 → 1.767 s for 32 tokens (14B). GPU time showed no clear improvement and
  background load varied, so these results do not establish higher decoding
  throughput.
  Greedy decode now reduces vocabularies of at least 16384 entries in parallel
  4096-entry blocks, then merges their winners. Repeated penalties, non-finite
  handling and lowest-index ties retain the previous behavior. Partial winners
  use eight bytes per block (256 bytes for 131072 entries), owned by each decoder;
  mapped weights and shared KV are unchanged. Three paired Ministral-3B runs on
  the M2 Max saved a median 133–219 µs per complete decode step, with identical generated
  tokens and residual/KV bits. Complete cached requests gained 0.55–3.20% per
  paired run, reaching 75–76 tokens/s in the faster runs and 59 under heavier
  host load. These used a warmed 27-token chat prompt and 64-token output cap;
  the longer fixed-input benchmark below is a different workload. See
  [all measurements and reproduction commands](../benchmarks/metal/ministral-decode.md).
  For at most 128 visible positions, attention tiles and their ordered merge
  now share one threadgroup. Three cold-prompt Ministral-3B comparisons measured
  a small 0.24–0.80% decode gain with unchanged KV and mapped-weight storage.
  Qwen's hybrid decoder overlaps its direct Q6_K QKV projection with independent
  CPU projections and reuses the gate slab for SiLU, saving 2.125 MiB of shared
  activation scratch at 32 positions. Its projection latency improved, but
  paging and host-load variation prevented a stable whole-model speedup estimate.
  See the [32 GiB benchmark report and raw data](../benchmarks/metal/decode-paging.md)
  for timings, memory counters, limitations and opt-out switches.
  To check full logits, greedy predictions, residual/KV bits and weight ownership
  against fixed inputs with a local model, run:

  ```sh
  GOPHERLLM_MINISTRAL_MODEL=/path/to/Ministral-Q4_K_M.gguf \
    CGO_ENABLED=1 go test -tags metal -run '^TestMinistralConfiguredDecode$' -count=1 -v .
  ```

  Add `GOPHERLLM_MINISTRAL_AUTOTUNE=1` to exercise quick calibration, including
  the native row-group sweep, without writing a persistent tuning cache.

  This uses eight threads, F32 KV, 160 prompt tokens and 64 decode inputs.
  `GOPHERLLM_MINISTRAL_DECODE_CONTEXT`, `GOPHERLLM_MINISTRAL_DECODE_STEPS` and
  `GOPHERLLM_MINISTRAL_DECODE_REPS` adjust the workload.
  `GOPHERLLM_MINISTRAL_DECODE_SWEEP=1` interleaves 4/2/8/8/2/4 SIMD-group counts
  in one loaded model and requires identical output bits across all schedules.

## CPU kernels

- ARM64 Q8_0 prefill processes four tokens together while preserving the
  existing Q8 arithmetic. Native dense Gemma 4 uses layer-wise prompt batches,
  including its token-dependent per-layer projections. On the measured M2 Max,
  these additional changes reduced TTFT by 58–59% for Qwen 4B Q8_0 and 24–26%
  for Gemma 4 E2B Q4_K_M, with identical text in the comparison runs.
- ARM64 activation quantization uses a baseline NEON kernel with the portable
  implementation's rounding and exceptional-value behavior. A startup parity
  check can disable it, and `GOPHERLLM_DISABLE_SIMD=1` selects the portable
  reference. Isolated quantization measured 7.3–8.0× faster on the M2 Max with
  no allocations; complete projection and model gains are smaller.
- Q8_0 Q/K/V and gate/up projection groups share activation quantization and
  dispatch through the existing int8 kernels. Native Gemma also groups compatible
  CPU Q4_K/Q6_K attention and dense FFN projections, preserving its shared-KV
  behavior. On the measured M2 Max, Qwen 4B Q8_0 decode improved by 2.3–2.4×;
  Gemma 4 E2B Q4_K_M decode improved by 5–6% and TTFT fell by 7%. Qwen output
  can change because more projections now use Q8 activation approximation;
  `GOPHERLLM_Q8_ACTIVATIONS=0` retains float fusion.
- On x86-64 (AVX2 + FMA + F16C, auto-detected via CPUID), Q4_K, Q5_K, Q6_K,
  Q8_0, Q4_0, Q4_1, MXFP4, Q2_K, and Q3_K matvecs default to int8-activation full-row kernels: the activation
  vector is quantized once per matvec to int8 with one scale per 256-element
  block (llama.cpp's Q8_K convention, `q8kQuantize`), and each weight row is
  processed by a single assembly call (`q4kDotQ8KRow` / `q5kDotQ8KRow` /
  `q6kDotQ8KRow` / `q8_0DotQ8KRow`) that decodes block scales in-register,
  dots 32 weights per `VPMADDUBSW` (Q8_0's own signed weights use the
  abs/sign-restore identity so the same unsigned-operand instruction applies),
  applies scales via `VPMADDWD`, and reduces horizontally once per row. Versus
  the previous per-block float kernels this is ~2.5x (Q4_K) to ~6x (Q6_K and
  Q8_0) per-row — and >20x for Q5_K, which previously had no SIMD fast path at
  all — and roughly 4x end-to-end decode on a Ministral 3B Q4_K_M. Set
  `GOPHERLLM_Q8_ACTIVATIONS=0` to force the exact float kernels
  (bit-reproducible against the scalar reference; the int8 path stays within
  cosine 0.999 of it — the same accuracy tradeoff llama.cpp makes by default).
  `GOPHERLLM_DISABLE_SIMD=1` disables the x86-64 SIMD dispatch and ARM64
  SDOT/activation-quantization fast paths; baseline ARM64 NEON float and
  attention kernels remain active.
  Q3_K is the newest of these and the one that gains most: its high bit is a
  per-*element* −4 bias rather than the per-sub-block constant the other
  K-quants carry, which is why it had no SIMD path for so long. It factors as
  `Σ(u + 4h)·a − 4·Σa`, keeping `u + 4h ≤ 7` inside `VPMADDUBSW`'s unsigned
  operand and costing one extra multiply-add against a vector of ones. Median
  of five interleaved runs on a 4096-column row: 54.8 µs scalar to 1.47 µs,
  about 37x.
- ARM64 int8-activation kernels require `SDOT` (`FEAT_DotProd`). Feature
  detection uses `/proc/self/auxv` and `HWCAP_ASIMDDP` on Linux/Android,
  `IsProcessorFeaturePresent` on Windows, and a feature sysctl on iOS.
  Apple Silicon Macs enable the path directly. Unsupported or undetected
  CPUs retain portable kernels. The f16 KV-cache and grouped-GQA f16
  attention kernels use baseline ARMv8.0 `FCVTL`/`FCVTN` conversion
  instructions, so they do not require the optional `FEAT_FP16` arithmetic
  extension or SDOT.
- Prompt processing (prefill) is batched. With the int8 path active, each raw
  quantized weight row is streamed from memory exactly once per prompt chunk and
  dotted against all prompt tokens' pre-quantized int8 activations in
  L2-resident row tiles (`matvecBatchQ8`) — no f32 dequantization pass at all.
  With `GOPHERLLM_Q8_ACTIVATIONS=0` the older dequantize-once-per-chunk f32 path
  runs instead. ARM64 reuses per-worker dequantization rows and dispatches one
  coarse batch range per worker to avoid allocation and scheduling overhead.
  The same path now covers StableLM's tensor-selected sequential or
  parallel-residual LayerNorm block, dense Qwen3's per-head QK norm, EXAONE
  4's QK plus post-branch norms, OLMo 2/3's full-projection QK plus post-branch
  norms, and Phi-2's shared biased LayerNorm plus parallel exact-GELU branch.
  This moves all five families' prompt ingestion from per-token weight
  streaming onto the chunked path as well. Q4_0/Q8_0 rows also use the
  dequantize-once path, which is important for Stable Code and other legacy-Q4
  GGUFs on Apple Silicon.
  Set `GOPHERLLM_NO_BATCH_PREFILL=1` to fall back to the per-token path (A/B
  benchmarking / debugging), or `GOPHERLLM_PREFILL_CHUNK=<N>` to tune the chunk
  size on the deployment machine.
- SwiGLU's `x*sigmoid(x)*up` runs through an AVX2 kernel with a Cephes-style
  expf polynomial (~1e-7 relative error) instead of per-element `math.Exp`.
- ARM64 SwiGLU evaluates four elements together using baseline NEON, with the
  scalar path's float64 range reduction, float32 polynomial and saturation
  bounds. Finite results match the scalar path bit for bit; short tails stay
  scalar, unequal slice lengths use their common prefix, and either input may
  be reused as the output. On an M2 Max, repeated single-thread kernel
  benchmarks measured 9,216 elements at 41.1 → 8.4 µs and 16,384 elements at
  72.8 → 14.6 µs (about 5× faster), with zero heap allocations. A three-round
  Ministral 3B CPU comparison retained identical greedy output. These timings
  measure the activation kernel; full-model timings varied too much to assign
  an end-to-end speedup.
- Float32 dot products on x86-64 CPUs without AVX2 use an SSE2 kernel with
  unaligned loads and the portable kernel's four-lane accumulation order.
  `GOPHERLLM_DISABLE_SIMD=1` also disables this fallback. Regression tests cover
  exact finite-result bits, unequal lengths, tails, unaligned slices and local
  stack arrays. The fallback has been exercised under Rosetta; native Intel/AMD
  performance has not been measured.
- ARM64 float32/float16 dot products and their four-head GQA variants reduce
  vectors with pairwise NEON additions while preserving the existing addition
  tree. On an M2 Max, controlled before/after benchmarks at 128 elements showed
  10–15% lower dot-kernel time and single-thread attention benchmarks showed
  1–8% lower time. These are kernel measurements, not end-to-end model gains.
  Audited `noescape` declarations also keep local caller arrays on the stack
  for these float operations on ARM64 and dot products on x86-64.
- ARM64 CPUs with SDOT use full-row Q4_K, Q6_K and Q8_0 decode kernels.
  Prefill shares Q6_K unpacking across four tokens and packs Q8_0 activations
  once per batch for four-token SIMD dot products. Decode workers claim fine
  row partitions through a shared cursor, reducing channel dispatch overhead.
  These paths preserve the existing ARM64 arithmetic and have startup checks
  and fallbacks.
  Q6_K decode and four-token prefill use pairwise integer reductions to pack
  SDOT results without separate lane moves.
- On ARM64, Q4_K and Q6_K matvecs use NEON block kernels, attention heads are
  spread across the worker pool at longer contexts, and single-token matvec work
  is split into eight ranges per worker so performance cores absorb
  efficiency-core stragglers. This increased Ministral 3 14B decode from 6.67
  to 7.70 tok/s (+15%) in the local 48-token benchmark while remaining neutral
  within noise on Ministral 3 3B. The
  dispatch coordinator computes one shard itself instead of parking, and Q8
  activation scratch is returned without a per-projection closure allocation.
  The worker pool remains sized to the configured runtime even when a kernel
  exposes fewer independent shards, avoiding repeated 8/12-worker teardown and
  recreation between GQA and projection kernels. Four-query-head GQA (the
  32-Q/8-KV layout in Ministral 3B) has dedicated shared-row NEON dot and AXPY
  kernels: every K/V row is loaded once for all four queries. The isolated
  attention path measured about 2.1x faster at 4k and 1.9x at 32k context on an
  M2 Max. Decode dispatch retains the more parallel per-head path below 4k,
  where it won end-to-end, and switches to grouped GQA at 4k; the complete
  3B forward improved about 3-4% there. Batched prefill already has token-level
  parallelism and uses grouped GQA directly. Set `GOPHERLLM_NO_GROUPED_GQA=1`
  for an A/B rollback.
  For 128-channel F32 heads, weighted-V accumulation keeps 16 channels from
  all four queries in registers across 256-position tiles, preserving the
  chronological FMA sequence. That component measured 3.9x faster at 4k and
  4.5x at 32k positions on the M2 Max with Ministral's 1024-float V stride;
  these are kernel timings, not complete decoding speedups. Short spans and
  other long-context layouts retain the row kernel.
  Non-ARM64 targets (amd64 included) run the same grouped-GQA decode
  algorithm: the shared-row dot/AXPY step is composed from portable or x86-64
  SIMD `DotF32`/`AxpyF32` kernels instead of a hand-fused NEON kernel,
  but each K/V row is still read from the cache once per query-head group
  rather than once per head, so the same bandwidth saving applies. It reuses
  the 4k decode threshold above rather than a value re-tuned for amd64.
  Grouping is not limited to the 4:1 ratio or to the exact-f32 KV cache: any
  GQA/MQA layout with more than one query head per KV head goes through
  `attendHeadGroup`, which picks the dedicated 4-head kernel when it applies
  and a generic shared-row fallback (`onlineAttentionGroupEither`) for every
  other ratio, and the f16/int8-block KV cache tiers have their own grouped
  kernels (`onlineAttentionGroupF16`/`onlineAttentionGroupI8`) so switching
  `GOPHERLLM_KV_F16`/`GOPHERLLM_KV_I8` no longer silently falls back to the
  ungrouped per-head path. This widens the shared-row win to the many
  non-4:1 GQA/MQA architectures (Falcon, StarCoder2, ChatGLM/GLM4, Command-R,
  and others) and, since f16 is the default KV cache on fast x86-64, to the
  default configuration generally — previously the grouped path only ever
  ran with `GOPHERLLM_KV_F16=0` forcing the exact f32 cache.
  ARM64 also uses NEON FP16 conversion, dot, and accumulation kernels
  for the optional compact KV cache; at 4k context they made the f16 attention
  benchmark about 3.7x faster than the scalar path. Exact f32 remains the
  default there because it was still faster on the measured M2 Max. These were
  Apple-only until the `FCVTL`/`FCVTN` kernels were un-gated from `darwin`;
  they are baseline ARMv8.0, so every arm64 target gets them.
- Set `GOPHERLLM_DISABLE_YARN=1` to skip YaRN RoPE scaling for models that declare
  it.

## KV cache and paging

- Split GGUFs (llama.cpp's `gguf-split` naming convention,
  `<name>-00001-of-00005.gguf`) are detected from any one shard's
  `split.count` metadata; every sibling is located next to it, and their
  tensor data is merged into one in-memory buffer before loading. This costs
  one full copy of the model's weights at load time. Use `--out-of-core`
  to map shards separately without that merged allocation, as described above.
- On x86-64 (F16C) the KV cache stores K/V rows as f16 by default: half the
  cache memory (double the context fits the reusable-workspace cap) and half
  the bytes attention streams per generated token, with rows converted
  in-register (`VCVTPH2PS`) inside the attention kernels. Greedy decode on
  the test model is bit-identical to the f32 cache; set `GOPHERLLM_KV_F16=0`
  to force the exact f32 cache. Attention itself is two-pass (independent
  score dots, then max-stabilized softmax weights and the weighted V
  accumulation), which measured ~1.15x over the previous online-softmax loop
  at 4k-16k context and uses the true score maximum for stability.
  On non-amd64 systems the exact f32 cache remains the default; set
  `GOPHERLLM_KV_F16=1` to opt into the compact cache when memory capacity
  matters more than decode speed (for example, large Kimi contexts on a
  unified-memory Mac). ARM64 converts its rows with dedicated NEON
  FP16 kernels; other non-amd64 targets use the portable scalar fallback.
- A third, more aggressive KV cache tier stores K/V rows as Q8_0 blocks
  (`GOPHERLLM_KV_I8=1`, off by default everywhere): one f16 scale plus 32
  signed int8 values per 32 elements, the same row format this project's
  `--compress` tool already writes for weight tensors. This is a
  **memory-capacity feature, not a speed feature** — be clear-eyed about the
  byte math before enabling it:

  | format | bytes/element | vs. f32 | vs. f16 |
  |---|---|---|---|
  | f32 | 4 | 1x | — |
  | f16 | 2 | 2x | 1x |
  | int8 (Q8_0) | 1.0625 | 3.76x | 1.88x |

  f16-over-f32 is a clean 2x, which is why it's the amd64 default; int8's
  *incremental* saving over the already-default f16 cache is much smaller
  (1.88x, not another 2x). At typical short chat context (dozens to a few
  hundred tokens) that difference is a small fraction of a percent of total
  per-token memory traffic — most of decode's cost is the weight matvecs, not
  the KV-cache scan — so expect **no measurable decode-speed change** there.
  The saving only starts displacing real DRAM traffic once resident context
  is large (roughly the same multi-thousand-token regime that already
  justifies this project's grouped-GQA threshold), where it can fit
  meaningfully more context in the same memory budget than f16 alone. Only
  enabled when every relevant dimension (`K`/`V` width and each individual
  head's width) is a multiple of 32 — Q8_0 rows are only addressable at block
  boundaries, so a model whose head dimension isn't 32-aligned automatically
  falls back to the f16 or f32 cache instead of silently corrupting attention.
  Scalar-only for now (no AVX2/NEON kernels yet); not part of `--auto`'s
  search.
- After mmap'ing a single-file GGUF, the CLI warms backbone tensors across
  worker threads before reporting the model loaded (`--prefault core`).
  Packed sparse expert banks, untied packed input embeddings and Gemma 4's
  per-layer token lookup tables remain demand-paged. `--prefault all` touches every page up front, matching the
  library's zero-value load policy.
  A memory-mapped file only pages in on first touch. A dense forward pass
  touches essentially every weight byte — without warm-up, the *first*
  request after startup silently inherited that page-in cost (disk I/O, or on
  Windows, real-time antivirus scanning of each mapped page) inside its own
  TTFT instead of load time. For a one-shot CLI run this doesn't change total
  wall-clock; for a REPL or another long-running inference process it makes
  first-use latency predictable instead of leaving one request to absorb a
  multi-second page-in tax. Set `GOPHERLLM_NO_PREFAULT=1` to restore pure lazy
  paging.

## M2 Max, M4 and M4 Pro Macs

The CPU mix and memory interface differ substantially between these chips:

| Mac chip variant | Performance + efficiency cores | GPU cores | Peak memory bandwidth |
|---|---|---|---|
| M2 Max | 8 + 4 | 30 or 38 | 400 GB/s |
| M4, 10-core CPU | 4 + 6 | 10 in Mac mini | 120 GB/s |
| M4 Pro, 12-core CPU | 8 + 4 | 16 | 273 GB/s |
| M4 Pro, 14-core CPU | 10 + 4 | 20 | 273 GB/s |

Sources: [Apple's M2 Max specifications](https://support.apple.com/en-us/111835),
[Mac mini specifications](https://support.apple.com/en-us/121555)
and [M4 Pro announcement](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/).
The [two-port M4 iMac](https://support.apple.com/en-us/121556) instead has four
performance and four efficiency cores, an eight-core GPU and 120 GB/s bandwidth.

M4 adds SME/SME2 matrix acceleration with 512-bit streaming vectors
([primary research](https://arxiv.org/abs/2409.18779)). GopherLLM sends suitable
F32 prefill matrices to [Accelerate](https://developer.apple.com/documentation/accelerate),
which selects its matrix implementation for the available CPU at runtime.

- Auto mode measures the P-core count and a mixed count halfway toward all
  cores, alongside the existing generic candidates. M4 Pro's 12-core variant
  now includes 8 and 10 threads; the 14-core variant includes 10 and 12.
  These counts do not pin goroutines to particular cores; the tuner selects
  settings only after its timing and final-validation checks.
  Explicit thread counts remain candidates even when they are absent from the
  topology-derived list; tied measurements retain the current setting.
- Dense Metal decode also measures two, four and eight SIMD groups per
  threadgroup using the complete decode probe and final validation. The
  uncalibrated schedule uses four; explicit environment overrides take priority.
  The choice is cached with the model and host identity, allowing M2 Max and
  M4 to select different schedules without a chip-name assumption. Kernel
  arithmetic and mapped-weight ownership are unchanged.
- Auto-tuning and Metal-probe caches include chip name, Mac model, CPU family,
  P/E counts, exposed SME features and the availability of the CGO prefill
  backend. A result for a twelve-core M2 Max is no longer reused for a
  twelve-core M4 Pro. The Mac model also separates different chassis using
  the same chip. Unavailable queries fall back to the remaining platform
  information.
- With CGO, F32 prefill batches of at least 32 tokens use larger GEMMs instead
  of capping every call at 128 rows. Proven contiguous, disjoint input/output
  slabs can bypass activation copies. Other layouts retain input snapshots
  and the previous output-copy order. Quantized weight-expansion tiles keep
  their four-MiB limit; Go-side packed input and output each remain bounded
  by eight MiB. Short batches retain their previous compute tile size.
  The batch graph passes its owning slabs explicitly, so capacity-clipped
  token views also avoid packing. For 128 tokens and a 3,072 × 768 F32
  projection, this additional change measured 449 → 313 microseconds on the
  M2 Max, with zero allocations in both paths.

An isolated [SME int8 tile experiment](../benchmarks/sme/README.md) provides
opt-in hardware parity tests and benchmarks for an SME-capable Mac. It uses
SME1 SMOPA and reports SME2 availability; it does not change model inference.
Compilation and instruction inspection passed on the M2 Max, but its SME
hardware tests skip. M4/M4 Pro numerical and performance validation is still
required before integrating this experiment into prefill.

On the M2 Max validation host, three alternating A/B rounds for a
3,072-row × 768-column F32 projection with 64 tokens measured about **1.5×**
speedup for bounded per-token views and **2.4×** for contiguous slabs, with
zero steady-state heap allocations. These are projection measurements on
M2 Max; M4/M4 Pro timings have not been measured here. To measure the local
Mac and choose model-specific settings:

```sh
make run-auto MODEL="your-model.gguf" AUTO_EFFORT=balanced
CGO_ENABLED=1 go test -run '^$' -bench '^BenchmarkApplePrefillMatrixLayout$' -benchmem -count=5
```
