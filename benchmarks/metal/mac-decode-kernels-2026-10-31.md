# Dense Metal decode on the 32-GiB M2 Max

2026-10-31, Mac14,6, Apple M2 Max, 32 GiB, macOS 27.2, Go 1.27.3,
Metal + Accelerate, eight workers. Baseline: `f94a4e7`.

## Paired decoding

| Real checkpoint | Control median | Optimized median | Median paired gain | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Ministral 3 3B Instruct 2512 Q4_K_M | 71.91 tokens/s | 72.16 tokens/s | +0.85% | 16/20 |
| Ministral 3 14B Reasoning 2512 Q4_K_M | 20.84 tokens/s | 20.94 tokens/s | +1.69% | 16/20 |
| Voxtral Mini 4B Realtime 2602 Q6_K | 68.64 steps/s | 70.27 steps/s | +2.91% | 20/20 |

Each model is loaded once. After one warm request per variant, 20 pairs
alternate control/optimized and optimized/control. The paired gain is
`median(100 × (control_seconds / optimized_seconds − 1))`; it is not the
ratio of the two rate medians. Loading, compilation and prefill are excluded
from decode time. No builds or other project tests ran during measurements.

The primary control compiles the **exact pre-change shader source** in the
same benchmark executable. A measurement-only patch selects those original
Q4_K/Q6_K/Q8_0 projections when the control is requested. This avoids treating
the diagnostic pipeline in the new source as an exact code-generation match
for the old compiler input. The patch and executable hashes are in the
[JSON record](mac-decode-kernels-2026-10-31.json); it is not part of the runtime.
Two preceding 20-pair series using the built-in diagnostic control are also
included as supplementary measurements, not pooled into the table.

Ministral uses a 27-token German prompt, 64 autoregressive output tokens,
temperature zero, seed 41 and repeat penalty 1.1. Warm requests reuse the
same prompt and boundary logits. Voxtral loads the complete real checkpoint
and runs its full decoder with 32 teacher-forced prefix embeddings, followed
by 64 fixed embedding inputs with vocabulary selection. Voxtral steps/s
measure decoder work, not full audio transcription latency or recognition
quality, and are not directly comparable with chat tokens/s.

All output-token digests match within each workload, including fresh-process
stock/candidate runs. The gains are small and specific to this workload and
host. Other applications remained active; independent-process timings vary
more than these paired gains. There is no claim of the same gain on M4 or
other Apple GPUs, or of reduced paging under general memory pressure.

## Kernel changes

Q4_K projections keep independent low/high FP32 sums, shortening the serial
accumulation chain. This changes floating-point addition order. Complete
synthetic decoder states stay within the existing GPU/CPU tolerance (maximum
scaled difference about 1.66e-4 in the tested fixtures); real-model token
sequences match for the recorded workloads. Set
`GOPHERLLM_METAL_DENSE_Q4_SPLIT_SUM=0` to retain the previous addition order
in dense single-token decoding.

Matching Q6_K gate/up projections share input loads and write the SiLU product
in one dispatch, extending the existing Q4_K fusion. This removes two
dispatches and one intermediate barrier per Q6 FFN. Regression tests preserve
residual, normalized and newly written KV values bit for bit, including
streaming compaction. `GOPHERLLM_METAL_DENSE_FUSED_GATE=0` restores separate
projections and activation. Mixed gate/up formats keep the separate path.

Q4 accumulation and Q6 unpacking are shared helpers. The runtime adds two
fixed diagnostic pipelines, with no extra weight or KV buffers.

## Memory and paging

Fresh processes run stock/candidate/candidate/stock for each model. The
following are peak process RSS ranges, including loading and prefill; they
are not total unified GPU-memory usage.

| Checkpoint | Stock peak RSS | Optimized peak RSS |
| --- | ---: | ---: |
| Ministral 3 3B Instruct 2512 Q4_K_M | 2.069–2.069 GiB | 2.068–2.069 GiB |
| Ministral 3 14B Reasoning 2512 Q4_K_M | 7.390–7.393 GiB | 7.393–7.393 GiB |
| Voxtral Mini 4B Realtime 2602 Q6_K | 4.464–4.464 GiB | 4.464–4.464 GiB |

The [earlier memory savings](mac-model-memory-2026-10-31.md) remain:

| Oversized Ministral context | Live Go heap after request + GC | KV bytes retained by the temporary Metal decoder |
| --- | ---: | ---: |
| Ministral 3 3B Instruct 2512 Q4_K_M (2,552 prompt tokens) | 58.7–58.7 MiB | 0 |
| Ministral 3 14B Reasoning 2512 Q4_K_M (1,670 prompt tokens) | 67.1–67.2 MiB | 0 |

GC runs after each oversized request, outside the timing windows. These
heap measurements do not establish a corresponding physical RSS reduction.
Long-context output digests also match the previous memory-optimized runs.
Voxtral still starts with 256 KV positions (52 MiB logical storage), borrows
134,184,960 bytes for additional decoder weight handles and copies zero bytes
for those handles. These counters exclude pre-existing prepared weights and
the encoder’s F32 storage.

The primary paired runs report these Mach task page-in events across the
measured requests (including decoder setup/prefix/teardown for Voxtral):

| Checkpoint | Control page-ins | Optimized page-ins |
| --- | ---: | ---: |
| Ministral 3 3B Instruct 2512 Q4_K_M | 0 | 0 |
| Ministral 3 14B Reasoning 2512 Q4_K_M | 5 | 11 |
| Voxtral Mini 4B Realtime 2602 Q6_K | 4 | 2 |

Mach page-ins are VM events, not counts of SSD reads, and do not measure all
GPU paging. The JSON also records global swap/page-in counters before/after
each process and selected counters alongside sampled RSS. They include other
applications. Swap occupancy is not active paging, and these samples do not
establish reduced swap I/O. Mach resident/footprint counters, RSS, live heap
and logical GPU capacities describe different accounting views and must not
be added together.

## Reproduce

Build the normal harness and set a local checkpoint path:

```sh
go test -c -tags metal -o /tmp/gopherllm-engine.test ./internal/engine
cd internal/engine
GOPHERLLM_REAL_MAC_MODEL=/path/to/model.gguf \
GOPHERLLM_REAL_MAC_KIND=ministral \
GOPHERLLM_REAL_MAC_PAIRS=20 \
/tmp/gopherllm-engine.test -test.run '^TestRealMacDecodeKernels$' -test.v
```

Use `voxtral` for the realtime checkpoint. This normal build uses the built-in
diagnostic control. To reproduce the primary comparison, apply `control_patch`
from the JSON to a disposable checkout before building, then restore the
header. That patch changes only the benchmark control pipelines.

For independent-process RSS and global VM sampling, use
`benchmarks/metal/measure-real-mac.py --binary /tmp/gopherllm-engine.test
--model /path/to/model.gguf --kind ministral --output /tmp/fresh-run`.
Set `GOPHERLLM_REAL_MAC_LONG_CONTEXT=1` for the oversized-cache check.

Validation: complete native Metal/Core ML tests, portable `CGO_ENABLED=0`
tests, native `go vet`, generated API check and the new state-comparison
tests. No model files or generated test binaries are committed.
