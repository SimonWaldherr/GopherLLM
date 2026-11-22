# Native LittleBit inference

GopherLLM runs LittleBit and LittleBit-2 projections in Go using the packed factors. The supported text decoders are dense Llama (GGUF `llama`, `llama2`, `llama3`) and dense Qwen3 (`qwen3`). Attention Q/K/V/O, SwiGLU gate/up/down, and a separate vocabulary head can use factors alongside normal GGUF weights. Token embeddings stay in a row-readable encoding. When `output.weight` and its factors are absent, the standard embedding weight serves as the output head.

## Convert and run

The converter accepts already trained Hugging Face checkpoint directories with `config.json` and `model.safetensors`, or a sharded checkpoint with `model.safetensors.index.json`. It reads Samsung-style packed I32 factors and their I64 shape tensors, I8 signs (−1 and +1), or trained F16/BF16/F32 factors using the binary forward sign convention (zero maps to +1). It also reads the learned scales, residual branch, ordinary weights, tokenizer and chat template. The Llama conversion reorders Q/K output channels to match the GGUF RoPE layout. Qwen3 retains its checkpoint ordering. The converter infers the variant from `use_itq`. For an older checkpoint with no such setting, pass `-variant 1` or `-variant 2`.

```sh
go run ./cmd/gopherllm-convert-littlebit -in /path/to/checkpoint -out /path/to/model.gguf
go run ./cmd/gopherllm /path/to/model.gguf --analyze
go run ./cmd/gopherllm /path/to/model.gguf --temperature 0 --max-tokens 32 --prompt "Why is the sky blue?"
go run -tags metal ./cmd/gopherllm /path/to/model.gguf --metal --temperature 0 --max-tokens 32 --prompt "Why is the sky blue?"
```

The converter and runtime use the Go standard library. The Metal build additionally uses cgo and the Apple frameworks. The importer accepts safetensors. PyTorch pickle files can be exported to safetensors offline while retaining the trained tensors and names. The converter supports Llama SentencePiece BPE (`tokenizer.model` or byte-fallback `tokenizer.json`) and Llama/Qwen3 byte-level BPE (`tokenizer.json`).

Samsung's [LittleBit implementation and documentation](https://github.com/SamsungLabs/LittleBit) are CC BY-NC 4.0. GopherLLM's implementation uses the published mathematics and verifies file compatibility with the reference. Each checkpoint retains its own license terms. The public [LittleBitLLM Qwen3-8B 0.55 BPW checkpoint](https://huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw) is an early LittleBit-2 distillation checkpoint under CC BY-NC 4.0, with the Qwen3 base under Apache 2.0. The validation fixture uses revision `6b2c47aa9707e624e2e737d16d64a66be8090ae9`.

Training is separate. LittleBit initializes factors with SVD-based decomposition and recovers quality through quantization-aware training or distillation. LittleBit-2 adds internal rotation and Joint-ITQ before training. Rotation is already incorporated into the exported factors. The conventional `--compress` row-wise quantizer has no LittleBit target and rejects existing LittleBit GGUF files. [LittleBit paper](https://papers.nips.cc/paper_files/paper/2025/file/a917c1d57088897beba47f96b4fd7f3c-Paper-Conference.pdf) [LittleBit-2 paper](https://proceedings.mlr.press/v306/lee26y.html)

## GGUF composite projection schema v1

Use a standard little-endian GGUF v3 container and the ordinary architecture, geometry and tokenizer metadata. The schema uses the registered GGML `I32` (26), `F32` (0), `F16` (1), and `BF16` (30) tensor IDs. Consumers need schema-aware projection loading.

| GGUF metadata key | Wire type | Value |
| --- | --- | --- |
| `gopherllm.littlebit.version` | UINT32 | `1` |
| `gopherllm.littlebit.variant` | UINT32 | `1` for LittleBit, `2` for LittleBit-2 |
| `gopherllm.littlebit.original_parameters` | UINT64 | Sum of `out × in` for all factorized projections |
| `gopherllm.littlebit.effective_bpw` | FLOAT64 | `8 × payload_bytes / original_parameters` |

The payload sum includes the stored I32 words (including unused bits in each row), four vectors per branch and any factorized biases. It excludes GGUF alignment gaps, headers, the tokenizer, and ordinary tensors. Use the sizes on disk even when inference accumulates in FP32. The whole-model tensor BPW also counts ordinary tensors and their parameters. A 0.1 BPW target refers to the original projection weights and can differ from the actual stored BPW.

Let `P` be the normal GGUF projection name, such as `blk.0.attn_q.weight`. Emit the following entries in place of the single dense tensor named `P`. GGUF dimensions below are fastest-varying first.

| Tensor | Type | GGUF dimensions |
| --- | --- | --- |
| `P.lb.u` | I32 | `[ceil(rank/32), out]` |
| `P.lb.v` | I32 | `[ceil(in/32), rank]` |
| `P.lb.u1` | F16, BF16, or F32 | `[out]` |
| `P.lb.u2` | F16, BF16, or F32 | `[rank]` |
| `P.lb.v1` | F16, BF16, or F32 | `[rank]` |
| `P.lb.v2` | F16, BF16, or F32 | `[in]` |
| `P.lb.bias` | F16, BF16, or F32 | `[out]`, optional |

Logical `U` is `[out, rank]` and `V` is `[rank, in]`. Pack each row on its own in increasing column order. In word `j/32`, bit `j%32` encodes column `j`. A bit value of 0 encodes +1 and a bit value of 1 encodes −1. Store each word little endian, including when bit 31 is set. Unused high bits in the last word must be zero. For example, negatives in columns 0, 2 and 31 of a 33-column row produce bytes `05 00 00 80 00 00 00 00`. Use standard GGUF tensor alignment.

To include a residual branch, add the complete set of six required entries under `P.lb.residual.`, with the same input and output dimensions and its own positive rank. Add the outputs of both branches before adding `P.lb.bias`. A factorized projection uses its composite bias entry exclusively. For ordinary projections, standard GGUF biases are supported where the decoder consumes them.

```text
z = V × (x ⊙ v2)
z = z ⊙ (v1 ⊙ u2)
y = (U × z) ⊙ u1
```

Use separate `attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `attn_output.weight`, `ffn_gate.weight`, `ffn_up.weight`, and `ffn_down.weight` for each block. Dense Qwen3 also carries ordinary `attn_q_norm.weight` and `attn_k_norm.weight` vectors. Fused QKV, fused gate/up layouts, MoE, factorized embeddings, and vision projectors are outside schema v1.

For split GGUF, put all four LittleBit metadata fields in shard zero. With ordinary split names and metadata, components of a projection can live in different shards. Disk-backed loading borrows each component from its own mapped shard when OS mmap is available. In-memory loads own a copy of the components. The loader validates required components, tensor types and shapes, finite scales, unused bits, byte ranges, and the declared parameter/BPW totals.

## Execution and repeatable checks

The portable kernel reads signs directly and uses floating-point signed sums. ARM64 NEON and x86-64 AVX2 handle full packed groups, with scalar tails and a worker pool for sufficiently large projections. Prefill can use batched activations. Metal keeps compact factor and scale buffers resident. The resident path processes a decode step or a prefill slab in one command buffer for eligible dense Llama and Qwen3 models with all seven projections factorized, 128-wide attention heads and RoPE, and an F32 KV cache up to 16,384 positions. An eligible mmap-backed BF16 output head also runs in the decode command buffer on Apple Silicon. Other layouts use the individual or grouped projections with the standard decoder. All paths operate on the stored factors directly.

```sh
go test ./...
go test -tags metal ./internal/engine ./internal/metal -run 'TestLittleBit'
GOMAXPROCS=4 go test ./internal/engine -run '^$' -bench '^BenchmarkLittleBit(Model|Matvec|ProjectionBaselines)$' -benchtime=300ms
GOMAXPROCS=4 go test -tags metal ./internal/engine -run '^$' -bench '^BenchmarkLittleBitMetal' -benchtime=300ms
```

For the published checkpoint, set `GOPHERLLM_LITTLEBIT_TEST_GGUF` to the converted GGUF. Each invocation runs three greedy requests with a cold prompt cache. On macOS, `time -l` reports process peak RSS in bytes.

```sh
go test -tags metal -c -o /tmp/littlebit.test ./internal/engine
GOMAXPROCS=4 GOPHERLLM_LITTLEBIT_TEST_GGUF=/path/to/model.gguf \
  /usr/bin/time -l /tmp/littlebit.test -test.run '^TestLittleBitPublishedCheckpoint$' -test.v
GOMAXPROCS=4 GOPHERLLM_LITTLEBIT_TEST_GGUF=/path/to/model.gguf GOPHERLLM_LITTLEBIT_TEST_METAL=1 \
  /usr/bin/time -l /tmp/littlebit.test -test.run '^TestLittleBitPublishedCheckpoint$' -test.v
GOPHERLLM_LITTLEBIT_TEST_GGUF=/path/to/model.gguf \
  /tmp/littlebit.test -test.run '^TestLittleBitPublishedCheckpointParity$' -test.v
```

## Measured results

Initial implementation on Apple M2 Max, Go 1.27.4, `GOMAXPROCS=4`. The public Qwen3-8B LittleBit-2 benchmark used a 26-token rendered prompt and 16 greedy tokens per run. The three CPU and three Metal runs returned the same 16 token IDs. The checkpoint echoed the start of the prompt in this run.

| Backend | Prefill tok/s, median (range) | Decode tok/s, median (range) | Process peak RSS |
| --- | ---: | ---: | ---: |
| ARM64 CPU | 6.36 (5.20–6.75) | 5.90 (1.90–6.00) | 5,531,041,792 B |
| Metal | 38.78 (33.08–38.92) | 6.01 (5.06–6.04) | 6,003,048,448 B |

- GGUF file: 2,974,395,904 B. Tensor payload: 2,968,248,576 B across 8,190,735,360 original parameters (2.89913 tensor BPW, 2.90513 file BPW).
- Factorized projections: 478,312,704 B across 6,945,767,424 original weights (0.55091128 stored BPW). Ordinary tensors: 2,489,935,872 B on disk. The runtime borrows 1,244,659,712 B of BF16 embeddings and widens the ordinary vocabulary head to 2,489,319,424 B of FP32 for these runs.
- Measured KV storage: 12,681,216 B for both runs. Metal held 478,312,704 B of factors and scales and 2,727,296 B of projection scratch. Those buffer lengths exclude driver overhead, CPU mappings, and other runtime allocations. Peak RSS is process-wide on Apple unified memory.

For implementation correctness, an independent NumPy FP32 forward read the original safetensors, unpacked each branch, and ran all 36 decoder blocks over token IDs `[16, 17, 18, 19]`. Tokenwise and batched CPU and Metal paths matched the reference's greedy token `13`. Across all 151,936 final logits, maximum absolute differences were `2.01e-5` for CPU and `7.87e-6` for Metal. On a separate three-token prompt, all three CPU positions also matched the reference's top-five order, with a maximum error of `2.83e-5`. Three real tokenizer probes matched Hugging Face. The opt-in `TestLittleBitPublishedCheckpointParity` checks golden projection outputs and final logits. Corpus perplexity remains unmeasured here. [Checkpoint evaluation](https://huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw)

The synthetic CPU benchmark uses a 4-layer Llama graph, hidden width 256, FFN width 768, fixed F32 embeddings, 32 prefill tokens and 16 decode steps. The conventional formats quantize the same dense projection reference. These runs measure execution and storage on the fixture.

| Projection format | GGUF bytes | Whole-model tensor BPW | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| LittleBit-2, rank 8 | 132,832 | 0.304 | 7,808 | 6,370 |
| Q1_0 | 475,584 | 1.198 | 6,793 | 322 |
| Q2_K | 1,065,408 | 2.694 | 8,359 | 2,711 |
| Q4_K_M mix | 1,920,960 | 4.865 | 16,516 | 4,454 |

The synthetic 4096 × 4096 projection benchmarks measured zero allocations per warmed call. On ARM64, rank 32 took about 62 µs with SIMD and workers, and rank 1,024 took about 342 µs. In the initial paired Metal measurements, single-activation dispatches took longer than CPU. Batch 32 improved from 3.53 ms on CPU to 0.39 ms on Metal at rank 32, and from 6.27 ms to 0.88 ms at rank 512 (medians). Those initial measurements used GPU projections with CPU attention and vocabulary projection.

### Qwen3-8B optimization pass

The BF16 vocabulary head stays mmap-backed on ARM64 and AVX2. The dot kernel widens each vector load to FP32 and uses the same accumulation and reduction tree as the widened head. On the released checkpoint this avoids a 2,489,319,424-byte FP32 copy. CPU prefill reuses each packed factor row across up to 32 tokens. On ARM64, NEON also shares each expanded sign mask across four tokens.

Metal puts Q/K/V in one command submission and the full SwiGLU sequence in another. Gate, up and the activated hidden vector stay on the GPU until the down projection finishes. With a separate attention output projection, a fully factorized decoder layer uses three synchronized submissions instead of seven. Mixed layers use compatible individual projection paths.

The opt-in comparison below runs the same greedy request with either the LittleBit checkpoint or the official Q4_K_M GGUF. It warms the runtime once, then clears the prompt cache before each of three timed generations. It reports the exact rendered-token hash, output hash, time to first token and file size. On macOS, `/usr/bin/time -l` reports peak process RSS. [Official Qwen3-8B GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf)

```sh
go test -tags metal -c -o /tmp/qwen3-bench.test ./internal/engine
GOMAXPROCS=4 GOPHERLLM_QWEN3_BENCH_METAL=1 GOPHERLLM_QWEN3_BENCH_GGUF=/path/to/qwen3-8b.gguf \
  /usr/bin/time -l /tmp/qwen3-bench.test -test.run '^TestQwen3LittleBitComparison$' -test.v
```

On M2 Max with four Go workers, both real models processed the same 26 prompt token IDs and generated 16 tokens. Rates are medians of three warm-runtime, cold-prompt runs. RSS is process-wide and in decimal GB. The before rows are measurements of the same LittleBit checkpoint and harness before the changes above.

| Configuration | Prefill tok/s | Decode tok/s | Peak RSS |
| --- | ---: | ---: | ---: |
| LittleBit CPU before | 6.28 | 5.84 | 5.54 GB |
| LittleBit CPU optimized | 12.17 | 6.22 | 3.05 GB |
| Q4_K_M CPU | 14.86 | 11.43 | 5.10 GB |
| LittleBit Metal before | 40.72 | 6.36 | 6.03 GB |
| LittleBit Metal optimized | 48.99 | 13.49 | 3.53 GB |
| Q4_K_M Metal | 77.87 | 34.24 | 5.08 GB |

All LittleBit runs returned the same 16 output token IDs. The optimized runtime was also checked against all 151,936 NumPy reference logits, with maximum absolute error of `1.04e-5` on CPU and `8.71e-6` on Metal. A separate 48-step Metal profile measured 72.59 ms/token in decoder layers and 14.53 ms/token in the BF16 vocabulary head.

### Resident Metal decode and BF16 head

The resident path encodes the decoder, BF16 vocabulary projection and greedy selection in one Metal command buffer per token. It uses the same GGUF mapping for the BF16 head and keeps the CPU KV cache authoritative for prefill, fallback and rewind. The projection shaders share input scaling across two latent rows and reuse each packed U word across four summands. The isolated, real Qwen3-8B BF16 head fell from 13.19 ms to 3.95 ms per greedy projection, with all 151,936 logits matching the CPU exactly for the benchmark input.

M2 Max and four Go workers, 26 input tokens and 16 generated tokens, medians of three runs. The two models used the same prompt token IDs and each kept its own greedy output. Disabling one path at a time measures its impact within this build.

| Metal configuration | Prefill tok/s | Decode tok/s | Peak RSS |
| --- | ---: | ---: | ---: |
| LittleBit before this pass | 48.68 | 13.99 | 3.53 GB |
| LittleBit, grouped projections and Metal BF16 head | 87.48 | 18.51 | 3.51 GB |
| LittleBit, resident decoder and CPU BF16 head | 85.03 | 28.47 | 3.53 GB |
| LittleBit, resident decoder and Metal BF16 head | 87.60 | 39.76 | 3.51 GB |
| Q4_K_M, same build | 78.79 | 38.83 | 5.08 GB |

For an A/B run, set `GOPHERLLM_METAL_LITTLEBIT_RESIDENT=0` or `GOPHERLLM_METAL_BF16_HEAD=0` before loading the model. The 16 generated LittleBit token IDs were identical in each configuration. With the resident decoder and Metal head, the final logits matched the independent checkpoint reference with maximum absolute errors of `8.71e-6` after prefill and `9.06e-6` after decode.

The short LittleBit run retained 478,312,704 bytes of packed Metal weights and 12,681,216 bytes of KV storage. The 1,244,659,712-byte BF16 vocabulary head uses the GGUF mapping directly. The reported peak RSS includes other runtime allocations and shared memory.

### Longer prompts

`TestQwen3LittleBitFixedDecodeComparison` uses the same benchmark environment with 128 prompt tokens and 128 fixed greedy decode steps. It measures inference without chat rendering or stop handling. Run it once per model with the same binary.

```sh
GOMAXPROCS=4 GOPHERLLM_QWEN3_BENCH_METAL=1 GOPHERLLM_QWEN3_BENCH_GGUF=/path/to/qwen3-8b.gguf \
  /usr/bin/time -l /tmp/qwen3-bench.test -test.run '^TestQwen3LittleBitFixedDecodeComparison$' -test.v
```

The Metal decoder uses slabs of up to 128 tokens by default when all seven packed projections in every layer have prepared Metal handles. `GOPHERLLM_PREFILL_CHUNK` overrides the slab size. The resident prefill path accepts eligible slabs of 16 through 256 tokens. Set `GOPHERLLM_METAL_LITTLEBIT_PREFILL=0` to use grouped prefill while keeping resident decode enabled. In a grouped Metal prefill sweep with the BF16 head on Metal, the mean of two timed runs after warm-up was 84.3 token/s with 32-token slabs and 91.6 with 128-token slabs for a 128-token prompt. For 512 input tokens, the corresponding rates were 76.0 and 88.1 token/s. Using 256-token slabs gave 86.9 token/s in that 512-token test. The argmax token agreed at every slab size for each prompt.

The same M2 Max with four Go workers produced these medians over three measured runs of the 128-step comparison.

| Metal configuration | Prefill tok/s | Decode tok/s | Peak RSS |
| --- | ---: | ---: | ---: |
| LittleBit, grouped 32-token slabs | 87.37 | 37.26 | 3.51 GB |
| LittleBit, grouped 128-token slabs | 96.07 | 36.74 | 3.51 GB |
| LittleBit, resident Metal prefill | 107.85 | 36.77 | 3.51 GB |
| Q4_K_M, same build as resident prefill | 146.02 | 34.74 | 5.08 GB |

On the 26-token prompt, the final LittleBit build reached 96.90 prefill and 38.76 decode token/s. Q4_K_M reached 78.06 and 38.00 token/s on the same build. In a separate A/B run, enabling resident prefill increased LittleBit from 84.43 to 99.21 prefill token/s and reduced the median time to first token from 308 to 262 ms. The two LittleBit paths returned the same 16 generated tokens, and the 128-step output IDs also agreed. The 128-token resident prefill retained about 68.5 MB of GPU scratch, including pending KV rows. Its 151,936 prompt logits agreed with grouped prefill to a maximum absolute difference of `9.54e-6`.

### Packed projection reuse in Metal prefill

LittleBit shares the packed V words and V scales across pairs of input tokens in batches of at least 16. Each token keeps its own FP32 accumulators, including for residual factors. The final pair in an odd batch writes one result. The Q4_K, Q6_K, and Q8_0 matrix kernels load four adjacent quantized values per lane and reuse their scale. The matrix tiles and accumulation order stay the same.

The M2 Max comparison with four Go workers, 128 input tokens and 128 greedy decode steps used the previous build at `c3ad12a` as the baseline. Each model returned the same greedy token IDs before and after. The figures are medians of three timed runs following a warm-up.

| Qwen3-8B checkpoint | Prefill before → after (tok/s) | Decode before → after (tok/s) |
| --- | ---: | ---: |
| LittleBit 2 | 111.01 → 142.77 | 37.59 → 36.55 |
| Q4_K_M | 148.52 → 204.93 | 36.33 → 36.03 |

With the 26-token prompt, the updated build measured 126.45 prefill and 38.87 decode token/s for LittleBit and 114.98 prefill and 38.25 decode token/s for Q4_K_M. Peak process RSS in the 128-step comparison stayed about 3.51 GB for LittleBit and 5.09 GB for Q4_K_M.

Four other cached models were checked on the M2 Max with 160 fixed input token IDs, 64 decode steps with full logits, and eight Go workers. Each cell is the median of three runs before and after the Metal change. All logits at the last decode step were bit-identical for each model.

| Cached model | Prefill before → after (tok/s) | Decode before → after (tok/s) |
| --- | ---: | ---: |
| Qwen3-4B Q8_0 | 342.67 → 420.99 | 64.22 → 63.58 |
| DeepSeek-R1-Distill-Qwen-7B Q4_K_M | 97.93 → 116.08 | 22.80 → 22.77 |
| Hermes-3-Llama-3.2-3B Q4_K_M | 353.38 → 492.33 | 77.09 → 77.76 |
| Ministral-3-3B-Instruct Q4_K_M | 328.94 → 453.08 | 74.22 → 74.44 |

```sh
GOPHERLLM_FIXED_MODEL=/path/to/model.gguf GOPHERLLM_FIXED_LOGITS=/tmp/final-logits.f32 \
  /tmp/qwen3-bench.test -test.run '^TestMetalFixedDecodeBenchmark$' -test.v
```

### Single-token Metal projection

For a single token with BF16 input scales and complete 32-column words, the Metal V projection reuses one scaled activation across four packed rows. Each row retains its own FP32 sum and reduction order. The generic kernel covers other scale types and partial words. This specialization applies to both branches when their V input scales meet the same conditions.

The regular Q4_K decoder fully unrolls the four fixed subgroups in each packed block. This makes the scale and minimum selection static in both the plain projection and the fused gate/up shader. In a separate 32-step Qwen3-8B control, all 151,936 final logits were bit-identical to the original loop.

On the M2 Max with four Go workers, both checkpoints generated the same token IDs before and after each change. Rates are medians of three warmed runs against the preceding prefill-optimized build.

| Qwen3-8B checkpoint | 26 input / 16 output (decode tok/s) | 128 input / 128 decode steps (tok/s) |
| --- | ---: | ---: |
| LittleBit 2 | 39.71 → 49.95 | 37.21 → 47.22 |
| Q4_K_M | 38.43 → 45.70 | 35.97 → 42.60 |

Prefill in the 128-token test measured 142.23 → 143.20 tok/s for LittleBit and 205.26 → 205.08 tok/s for Q4_K_M. The independent NumPy reference for the real LittleBit checkpoint agreed across all 151,936 logits with a maximum absolute decode error below `1e-5`.

The following test measures 48 greedy steps at each of several prefilled KV depths without charging prefill time to decode.

```sh
GOMAXPROCS=4 GOPHERLLM_QWEN3_BENCH_METAL=1 GOPHERLLM_QWEN3_BENCH_GGUF=/path/to/qwen3-8b.gguf \
  /tmp/qwen3-bench.test -test.run '^TestQwen3MetalDecodeContexts$' -test.v
```

The 48 generated IDs agreed before and after in every run at each depth. These are the medians of three runs per depth on the M2 Max.

| KV depth at first decode step | LittleBit 2 (tok/s) | Q4_K_M (tok/s) |
| ---: | ---: | ---: |
| 128 | 36.29 → 47.17 | 35.86 → 42.65 |
| 512 | 37.31 → 45.74 | 35.43 → 41.73 |
| 1,536 | 34.87 → 41.94 | 33.18 → 38.81 |

The same Q4_K change improved full-logit decode on the two cached 3B models measured with `TestMetalFixedDecodeBenchmark`. Hermes reached 89.31 from 77.75 tok/s and Ministral reached 88.89 from 74.46 tok/s. Both models had bit-identical logits at the last step.
