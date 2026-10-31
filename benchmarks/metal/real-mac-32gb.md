# Real checkpoints on a 32-GiB Mac

2026-10-30, Apple M2 Max, 32 GiB unified RAM, macOS, Go 1.27.3,
Metal + Accelerate, eight Go workers. Builds and tests ran outside measurement
windows. No model was downloaded or re-quantized for these measurements.

## Current decoding rates and memory

Each process loads one complete checkpoint. The first decode is excluded from
warm statistics; each table entry contains six warm 64-token samples.

| Checkpoint | Warm median | Warm range | Process peak RSS |
| --- | ---: | ---: | ---: |
| Ministral 3 3B Instruct 2512, Q4_K_M | 68.56 tokens/s | 60.66–72.25 | 2.069 GiB |
| Ministral 3 14B Reasoning 2512, Q4_K_M | 20.46 tokens/s | 19.81–21.19 | 7.391–7.392 GiB |
| Voxtral Mini 4B Realtime 2602, Q6_K, mapped loader | 68.23 steps/s | 65.24–74.84 | 4.831 GiB |
| Same Voxtral checkpoint, previous copying loader | 62.19 steps/s | 49.85–75.09 | 7.074–13.449 GiB |

Ministral runs full autoregressive chat requests with temperature zero, seed 41,
repeat penalty 1.1, F32 KV and core prefaulting. The German prompt in
`TestRealMacDecode` has 27 tokens and is fully cached for warm requests. The
64-token generation statistic includes the first token from boundary logits;
loading and prefill are excluded from its decode time. Each checkpoint was
loaded in two separate processes. Token digests match across all repetitions.
These are current rates, not proof of a new Ministral decoding speedup.
These short contexts do not establish throughput at thousands of KV positions.

Voxtral loads the entire real model, including its prepared encoder weights,
and runs its full Metal decoder. Inputs are 32 fixed token embeddings followed
by 64 fixed embedding inputs with complete vocabulary selection. No synthetic
weight matrices are used. This isolates decoding; it does not measure audio
encoding, complete transcription latency, or recognition quality. It is not
directly comparable to autoregressive chat throughput. All nine output-token
digests per loader agree, including between loaders.
Audio encoding can require additional workspace, so this is not a peak-memory
measurement of a complete STT request.

## Voxtral memory change

The previous file loader copied quantized tensors into Go-owned byte slices and
then into Metal buffers. `OpenVoxtral` and the file transcription entry point
now borrow mapped tensor storage through the existing no-copy Metal path.
The public bytes loader still copies, and the public standalone weight loader
retains its owned-storage behavior. Model/session ownership keeps file mappings
alive until all GPU users and weights have been released.

Across three processes per loader, loaded Go heap falls from approximately
**7.04 to 3.65 GiB**. The first comparison was copy/mapped; the repeat order was
mapped/copy/copy/mapped. Mapped peak RSS stays near 4.83 GiB; copying varies
between 7.07 and 13.45 GiB as residency and memory pressure change. The RSS
reduction is clear, but throughput ranges overlap substantially. No guaranteed
decoding speedup is inferred from these runs.

RSS is process residency measured by `ps` and Darwin `getrusage`, not complete
accounting of all unified GPU/driver memory. Go heap numbers describe only Go
allocations. RSS can include shared mapped pages and vary as the OS evicts
them. Do not add these counters together or treat them as total system RAM.

## Paging observations

Before the first run, about **5.5 GiB of swap was already occupied** by the
running system. Applications, rendering and thermal conditions were not
controlled. No file-cache purge was performed; first-load times are not
guaranteed cold-SSD measurements.

The first Ministral-14B run observed **130,656 system-wide swap-out pages**;
the initial copying Voxtral run observed **265,624**. Repeat Ministral runs and
all three mapped Voxtral runs observed zero swap-out pages during their
measurement windows. Later copying runs also observed zero, after earlier
memory pressure had already changed system residency. These counters cannot
be attributed exclusively to the benchmark processes.

Warm Ministral decoding adds only 5–22 process major-fault counter increments
per run; mapped Voxtral warm decoding adds zero in all three runs. Weight
loading and first use account for most process faults. A mapping can fault
pages at first decode rather than at load time. Neither process fault counts
nor `vm_stat` page-ins imply swap I/O alone: file-backed weights also cause
faults/page-ins. Swap occupancy remaining high after a run is not proof of
active decoding-time paging. Page-count conversion describes virtual page
volume, not necessarily compressed bytes written to disk.

## Validation and reproduction

Full native Metal/Core ML and portable Go suites pass, as do `go vet` and the
generated API check. An opt-in real-checkpoint ownership test verifies mapped
embedding storage, an active Metal session surviving model Close, and final
mapping release after session Close. The review on 2026-10-31 also covers
failed loads in partially populated encoder/decoder layers. Historical samples
and measured-source hashes remain unchanged; the JSON records the subsequent
error-cleanup patch and reviewed hashes. No new speedup is attributed to it.

```sh
CGO_ENABLED=1 go test -tags 'metal coreml' -c -o /tmp/engine.test ./internal/engine
python3 benchmarks/metal/measure-real-mac.py \
  --binary /tmp/engine.test --kind ministral \
  --model /path/to/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
  --output /tmp/ministral3b-measurement
```

Repeat for 14B. For Voxtral use `--kind voxtral` with the Q6_K checkpoint;
add `--copy` to select the old loader as a control. Use fresh output directories
per process, alternate loader order, and keep builds outside timing windows.
The sampler reads RSS and global VM counters every 0.5 seconds plus collection
overhead; short peaks may be missed by sampling, so process maximum RSS is
also retained. Run from a host permitting Metal, `ps`, `vm_stat` and `sysctl`.

[All ten runs, raw phase records, sampled counters, limits and source/binary hashes](real-mac-32gb-2026-10-30.json).
