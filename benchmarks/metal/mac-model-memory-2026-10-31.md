# Ministral and Voxtral memory on a 32-GiB Mac

2026-10-31, Apple M2 Max (Mac14,6), 32 GiB RAM, Go 1.27.3,
Metal + Accelerate, eight workers. Baseline: `ed0a699`.
The checkpoints are the same cached Ministral 3B/14B Q4_K_M and Voxtral
Mini 4B Realtime Q6_K files used in the [earlier model runs](real-mac-32gb.md).

## Short-context decoding

| Checkpoint | Baseline warm median | Changed warm median | Baseline peak process RSS | Changed peak process RSS |
| --- | ---: | ---: | ---: | ---: |
| Ministral 3 3B Instruct 2512 | 75.30 tokens/s | 75.68 tokens/s | 2.068–2.069 GiB | 2.069 GiB |
| Ministral 3 14B Reasoning 2512 | 21.00 tokens/s | 21.96 tokens/s | 7.393 GiB | 7.393 GiB |
| Voxtral Mini 4B Realtime 2602 | 77.93 steps/s | 78.37 steps/s | 4.714–4.831 GiB | 4.464 GiB |

These measurements establish a Voxtral RSS reduction. They do not establish a
sustained decoding speedup: ranges overlap, one baseline 14B process was slower
than its repeat, and host activity was not controlled. A fused Q6 projection/
argmax experiment and row-group scheduling probes did not establish a reliable
improvement and were left out of the runtime changes.

Ministral runs complete 64-token autoregressive requests with temperature zero,
seed 41 and repeat penalty 1.1. The 27-token prompt and boundary logits are
cached for warm repetitions. Each variant has two processes and six warm
samples. First requests and load times remain in the JSON but are excluded from
warm summaries. No file-cache purge was performed: a fresh process does not
imply a cold SSD read.

Voxtral loads the complete real checkpoint, including prepared encoder weights,
and measures its full Metal decoder with 32 teacher-forced token embeddings
followed by 64 fixed embedding inputs with vocabulary selection. Each variant
has four processes and eight warm samples. This measures real decoder weights,
not full audio transcription latency or recognition quality. It is not directly
comparable to autoregressive chat throughput.

Both comparisons use baseline/changed then changed/baseline ordering. All
repetitions and variants have identical output-token digests per workload.
Builds and tests ran outside measurement windows.

## Releasing large Ministral KV caches

A request above the 512-MiB reusable KV budget correctly avoided storing its
cache in `Runner.workspaceCache`, but the reused Metal decoder still retained
that cache through its pointer and pinned rows. It could therefore remain live
until another inference or model Close. Generation and decoder embeddings now
release the Metal state for a temporary cache at request exit, including errors
and cancellation. Retained small workspaces keep their decoder and prefix reuse.

The additional workload grows the prompt just beyond that budget and generates
64 tokens. The 3B prompt has 2,552 tokens; 14B has 1,670. Four requests run per
process and variant. Prefix reuse is disabled for these oversized caches.
The control uses the same candidate code with only the two new deferred cleanup
calls removed through a Go overlay; its exact patch is in the raw data.

| Checkpoint | Baseline live Go heap after GC | Changed live Go heap after GC | Temporary KV retained by Metal, before → after |
| --- | ---: | ---: | ---: |
| Ministral 3B | 592.1 MiB | 58.7 MiB | 531.6 → 0 MiB |
| Ministral 14B | 612.2 MiB | 67.2 MiB | 542.2 → 0 MiB |

`runtime.GC` runs after each request and before its heap snapshot, outside the
reported prefill/decode times. There is no forced OS scavenging. This diagnoses
retention; applications retain their ordinary GC scheduling. These heap numbers
are not measured reductions in physical RSS or total unified GPU memory. Peak
process RSS in these long runs stayed near 2.069/7.393 GiB. Token digests match
between the control and changed versions. The long-context decode medians are
62.65 → 62.47 tokens/s for 3B and 20.77 → 20.52 for 14B; no speedup is claimed.

## Voxtral GPU storage

The fast decoder now borrows mapped bytes for additional Q/K/V/O projections
instead of copying them into private Metal weight buffers. Borrowing is enabled
only with file-mapping provenance; the model/session lease keeps the mapping
alive until all GPU users close. Bytes and safetensors callers retain owned
storage. The measured fast decoder borrows 134,184,960 bytes (about 128 MiB)
for its additional handles and copies zero weight bytes. This counter excludes
handles already prepared by the model and the F32 encoder.

The KV cache starts at 256 positions and grows with the stream, preserving all
live rows and absolute RoPE positions. It still retains the checkpoint's full
8,192-position attention window once needed. Growth replaces one layer at a
time to bound the intermediate allocation and preserves live rows on failure.
Final growth includes compaction headroom to avoid another full-history copy.

For the measured short workload, logical float32 KV buffer capacity falls from
8,448 positions / **1,716 MiB** to 256 positions / **52 MiB**. This is allocated
buffer capacity, not a claim that 1,664 MiB of physical RAM was released. RSS
and driver residency have different accounting; the observed process RSS
reduction is the one reported above. Long recordings will grow the cache.
The final two Voxtral process pairs validate the reviewed growth implementation;
earlier short runs stayed below the initial 256-position boundary. Numerical
tests cover multiple growth boundaries and subsequent compaction.

## Paging and validation

All 20 selected process windows observed **zero additional system-wide
swap-out pages**, in both variants. About 4.9 GiB of swap was already occupied;
Xcode and Chromium remained active. Thus these runs do not establish a swap-I/O
reduction. File-backed page-ins, process fault counters, swap occupancy and
active swap I/O are distinct measurements. Process RSS also excludes some
GPU/driver memory; do not add it to the Go heap or KV capacity figures.

Full native Metal/Core ML and portable Go suites pass, as do `go vet` and the
public API generation check. Targeted Metal tests verify temporary-cache
release, retained-workspace reuse and growing/compacting history against the
CPU decoder. The real Voxtral checkpoint ownership test checks borrowed weights,
model Close during an active session, and final mapping release.

## Reproduction

```sh
CGO_ENABLED=1 go test -tags 'metal coreml' -c -o /tmp/model-decode.test ./internal/engine
python3 benchmarks/metal/measure-real-mac.py \
  --binary /tmp/model-decode.test --kind ministral \
  --model /path/to/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
  --output /tmp/ministral3b-short
GOPHERLLM_REAL_MAC_LONG_CONTEXT=1 \
  python3 benchmarks/metal/measure-real-mac.py \
  --binary /tmp/model-decode.test --kind ministral \
  --model /path/to/Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf \
  --output /tmp/ministral14b-long --timeout 240
```

Repeat with both Ministral sizes. For Voxtral use `--kind voxtral` and its Q6_K
GGUF, without the long-context environment variable. Use fresh output directories
and alternate variant order. The long workload deliberately crosses the cache
budget; it is different from the short cached-prompt workload.

[All 20 runs, phase records, RSS/VM samples, control patch, summaries and source/binary hashes](mac-model-memory-2026-10-31.json).
