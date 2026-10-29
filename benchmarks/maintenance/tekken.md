# Tekken splitting and tokenization

Measured on 2026-10-29 on an Apple M2 Max with 32 GiB RAM, Go 1.27.3,
Darwin/arm64 and `GOMAXPROCS=8`.

The Mistral/Ministral/Voxtral Tekken splitter traverses UTF-8 byte offsets and
returns input slices. This removes the full rune buffer and per-piece string
copies. An uppercase run that cannot match the first word alternative is reused
for the second alternative. Token boundaries and malformed UTF-8 replacement
remain identical to the original rune-based implementation.

GPT-2-family byte mapping now reuses a call-local append-only builder instead
of allocating its backing storage for every piece. It resets after 64 KiB or
before a piece needs more reserve than remains. Ordinary reserve is capped at
64 KiB; a single large piece may exceed it, keeping its previous two-byte-per-
input-byte reservation. Public mutable byte maps are read on each call. No
persistent cache or new retained tokenizer state is introduced.

## Method

Seven fresh processes alternate order. Each sub-benchmark targets 100 ms.
`reference` uses the original rune splitter and per-piece mapping buffers;
`per_piece` uses the optimized splitter and original mapping buffers;
`workspace` uses both changes. Split benchmarks compare the original and
optimized splitter only. All encode modes use the same BPE merge code and
compare exact token IDs before timing.

Encode measurements use the real vocabulary and merge ranks from
`Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`. Only the GGUF header is parsed;
model tensors and neural inference do not participate. Inputs are deterministic
synthetic text in `BenchmarkTekkenOffsets`. `unique` contains 1,024 distinct
words; `long_unique` repeats that block eight times to cross slab boundaries.
`long_piece` places a 73,728-byte Japanese word after a short prefix.

## Splitter

| Corpus | UTF-8 bytes | Original → changed median | Median paired ratio | Allocations/op |
| --- | ---: | ---: | ---: | ---: |
| short | 31 | 0.59 → 0.38 µs | 1.47× | 7 → 1 |
| ascii | 1280 | 21.49 → 13.31 µs | 1.57× | 259 → 1 |
| unique | 9216 | 120.17 → 70.45 µs | 1.63× | 1026 → 1 |
| long_unique | 73728 | 945.77 → 528.52 µs | 1.79× | 8194 → 1 |
| long_piece | 73735 | 1464.67 → 799.89 µs | 1.90× | 4 → 1 |
| european | 2976 | 66.17 → 53.62 µs | 1.23× | 514 → 1 |
| digits | 1472 | 72.44 → 60.75 µs | 1.31× | 1028 → 3 |
| unspaced | 1920 | 48.99 → 26.46 µs | 1.89× | 3 → 1 |

## Complete encoding

The ratios below are medians of each pair's reference/changed duration. They
can differ from the ratio of the two displayed medians, especially for noisy
cases. The final column isolates the builder change by comparing `per_piece`
with `workspace` in each pair.

| Corpus | Original → changed median | Median paired ratio | Builder-only paired ratio |
| --- | ---: | ---: | ---: |
| short | 5.50 → 5.25 µs | 1.02× | 1.00× |
| ascii | 199.88 → 183.12 µs | 1.08× | 1.01× |
| unique | 1639.92 → 1489.60 µs | 1.13× | 1.02× |
| long_unique | 12807.19 → 12250.05 µs | 1.05× | 1.02× |
| long_piece | 18341.47 → 15565.29 µs | 1.04× | 1.01× |
| european | 657.09 → 627.19 µs | 1.02× | 1.03× |
| digits | 164.97 → 137.55 µs | 1.21× | 1.06× |
| unspaced | 455.40 → 408.09 µs | 1.01× | 1.05× |

Host activity produced substantial outliers, all retained in the raw data.
Short encoding has a builder-only paired ratio of about 1.00× with four of
seven pairs faster; these samples establish no short-prompt latency gain.
The isolated builder wins six of seven pairs for both unique-word sizes and
for digits, but the ordinary text gains are small. The large-piece case's
ratio of medians is much larger than its paired ratio, so it should not be
quoted as a stable 18% throughput gain. These are component measurements;
there is no generation, decoding, STT or full-model speed claim.

## Allocations and lifetime

| Corpus | Full original → changed bytes/op | Full original → changed allocations/op | Builder-only bytes/op |
| --- | ---: | ---: | ---: |
| short | 1448 → 1376 | 40 → 29 | 1400 → 1376 |
| ascii | 26032 → 16944 | 1511 → 997 | 18096 → 16944 |
| unique | 190976 → 127488 | 8039 → 5991 | 133632 → 127488 |
| long_unique | 1428930 → 937409 | 64256 → 47873 | 1002945 → 937409 |
| long_piece | 17453698 → 17330786 | 57385 → 57382 | 17265266 → 17330786 |
| european | 56384 → 38976 | 3272 → 2248 | 41792 → 38976 |
| digits | 67808 → 51680 | 2442 → 394 | 57824 → 51680 |
| unspaced | 284352 → 279616 | 1490 → 1488 | 279616 → 279616 |

The builder adds about 64 KiB of allocation in the large-piece control relative
to per-piece buffers: the short prefix receives a bounded slab before the large
piece is reserved. Overall allocation still falls relative to the original
rune-based path. The unspaced case already forms one piece, so the builder
change saves no allocations there. A consumed slab may remain referenced by
BPE symbols until they are overwritten or the call returns. These are
allocation totals, not peak live memory, RSS or paging measurements.

Returned pieces from `PreTokenizerTekken` can retain the input string while a
caller keeps them. A caller retaining a small piece of a larger prompt can use
`strings.Clone`. Ordinary encoding returns token IDs and retains no new state.

## Reproduction and validation

[Raw samples and provenance](tekken-m2-max-2026-10-29.json) contain source,
binary, input and checkpoint-header hashes, all timings and allocation counts.
Without the optional model path below, the benchmark uses a synthetic
vocabulary.

```sh
GOMAXPROCS=8 \
GOPHERLLM_MINISTRAL_TOKENIZER_MODEL=/path/to/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
  go test ./internal/tokenizer -run '^$' -bench '^BenchmarkTekkenOffsets$' \
    -benchtime=100ms -count=1 -v
```

Repeat seven times, alternating `TEKKEN_BENCH_REVERSE=0` and `=1`. Build once
with `go test -c` and run the binary for each sample to exclude builds.

Differential tests retain the original splitter and compare 6,000 seeded
Unicode/malformed-byte inputs plus boundary cases. Encoder tests cover generic
GPT-2, Tekken, Qwen and Kimi splitters, slab boundaries, a large piece after a
prefix, and custom/mutable maps containing missing bytes, NUL and invalid runes.

Package tests passed with cgo enabled and disabled, as did the race test. A
three-second, two-worker splitter fuzz run passed 34,300 executions.
