# Core ML embedding benchmarks

These optional checks compare `CPUOnly` with `CPUAndNeuralEngine` using a local
compiled BERT encoder and its source GGUF. Core ML chooses placement within the
permitted devices. Compute-plan counts describe planned placement, not observed
ANE utilization or an ANE-only timing.
An unavailable compute plan is reported separately; prediction errors and failed
source parity still fail the benchmark.

```sh
GOPHERLLM_COREML_GGUF=/path/to/encoder.gguf \
GOPHERLLM_COREML_BENCH_MODELS=/path/to/encoder.mlmodelc \
  CGO_ENABLED=1 go test -tags 'metal coreml' ./coreml \
  -run '^TestCoreMLConfiguredBenchmark$' -count=1 -v
```

`GOPHERLLM_COREML_BENCH_MODELS` accepts an OS path-list, separated by colons on
macOS. `GOPHERLLM_COREML_BENCH_UNITS` accepts `cpu,ane` (default), `ane`, or a
different order such as `ane,cpu`. For graph comparisons, repeat the model list
in reverse order. The harness closes each encoder before loading the next.
Set `GOPHERLLM_COREML_BENCH_PLAN=0` to skip optional compute-plan inspection
when that diagnostic service is slow or unavailable. Plan counters are then
unknown; compute-unit permission and prediction/parity checks remain active.

Each export is tested with 1, 4, 17 and 32 texts. Model loading and first-use
latency are reported separately. After three warmups, nine windows of at least
50 ms record individual request latencies, including tokenization, normalization
and result ownership. JSON lines include median, P90, throughput, token-slot
occupancy, device plan and Go allocations. Go heap measurements exclude native
Core ML allocations and are not a measurement of process RSS or device memory.

Results are checked against the source GGUF outside the timed calls. Token
counts must match and cosine similarity must be at least 0.995. Sequence lengths
are hard input limits: the harness and adapter never truncate to make a shorter
export fit. Use a sequence length that also covers real document/query prefixes
and boundary tokens.

The dependency-free adapter benchmark isolates Go preparation and output costs:

```sh
go test ./coreml -run '^$' -bench '^BenchmarkEmbedBatchAdapter$' -benchmem -count=5
```

Optional native graph tests compare the exporter with an independent NumPy BERT
implementation, including masking, CLS/mean pooling, batches and the last-layer
CLS optimization:

```sh
PYTHONDONTWRITEBYTECODE=1 GOPHERLLM_COREML_EXPORT_TESTS=1 \
  python -m unittest scripts/test_export_coreml_embeddings.py
```

The Python environment needs the exporter's optional Core ML conversion
dependencies. They are not GopherLLM runtime dependencies.

## M2 Max results, 2026-10-24

Host: M2 Max, 32 GiB unified memory, macOS 27.2, Go 1.25.3,
coremltools 9.0. Source: Granite Embedding 278m Multilingual Q4_K_M GGUF.
Background applications remained active. The workload repeats four short texts
with 7–16 tokens; loading, specialization and compute-plan inspection are
excluded. [Measurements](m2-max-2026-10-24.csv) include all four request sizes
and both forward/reverse export orders.

At 64 positions and batch size one, including the final exported model:

| Export / permitted devices | Median for one text | P90 for one text |
|---|---:|---:|
| FP16 / CPU only | 5.26 ms | 5.40 ms |
| FP16 / CPU + ANE | 3.18 ms | 3.23 ms |
| Palette8 + final CLS pruning / CPU + ANE | 1.75–1.77 ms | 1.78–1.84 ms |

The optimized query export reduced warm latency by about 45% relative to the
original CPU/ANE export. It still accepts the same 64-token input length.

The controlled compression sweep at 64 positions and batch size four:

| Export / permitted devices | Median for 32 texts | P90 for 32 texts |
|---|---:|---:|
| FP16 + final CLS pruning / CPU + ANE | 36.82–37.16 ms | 36.99–37.52 ms |
| Palette8 + final CLS pruning / CPU + ANE | 31.46–31.80 ms | 31.82–32.13 ms |

Palette8 reduced latency by roughly 14–15% relative to the otherwise identical
pruned FP16 graph. The `.mlpackage` shrank from 554.6 to 469.7 MB (decimal),
about 15%. Minimum cosine against the source on this timing corpus was
0.9998971 at batch size four and 0.9998955 in the final batch-one export,
versus at least 0.9999948 for the uncompressed ANE exports. This
approximation remains opt-in; validate it on the intended retrieval corpus.
The final exporter also passed a separate 12-document, eight-query multilingual
retrieval fixture: Top-1 agreement and Recall@3 against the GGUF were both 1.0,
with minimum embedding cosine 0.9999058 and maximum query/document score error
0.001957. Recall compares neighbour sets; close-score order can still change.

```sh
GOPHERLLM_COREML_GGUF=/path/to/encoder.gguf \
GOPHERLLM_COREML_RETRIEVAL_MODELS=/path/to/optimized.mlmodelc \
  CGO_ENABLED=1 go test -tags 'metal coreml' ./coreml \
  -run '^TestCoreMLConfiguredRetrievalParity$' -count=1 -v
```

```sh
python scripts/export_coreml_embeddings.py encoder.gguf optimized.mlpackage \
  --sequence-length 64 --batch-size 4 --prune-cls \
  --weight-compression palette8 --compile-to optimized.mlmodelc
```

Only sufficiently large convolution **weight** constants are palettized.
Vocabulary, bias and normalization values remain FP16. There is no activation
quantization. Use a new RAG `EmbedderID` when switching weight compression.

Shorter exports helped bulk requests: with the same four texts and CLS pruning,
32 positions took 32.96–33.04 ms and 16 positions took 30.90–31.72 ms for 32
texts. They accept fewer tokens, so these timings do not establish gains for
long documents. Individual queries still favor batch size one: the original
64-position export took about 3.18 ms per query and the optimized one about
1.76 ms. The final batch-four export measured 31.26–31.56 ms for 32 texts.
Unpruned batch-four FP16 runs varied from 37.69 to 41.23 ms across phases under
background load; quoted compression gains use the paired compression sweep.

Two alternatives were measured and rejected: combining separate Q/K/V weights
made this ANE workload slower (39.42–39.83 ms for 32 texts), and linear INT8
weights had lower source similarity and noisier bulk latency than palette8.
Neither alternative was added as an exporter option.

Fresh-process measurements with `/usr/bin/time -l` found approximately 138 MiB
peak RSS for FP16 and 140 MiB for palette8, with 91 page faults and zero process
swaps in both. Those processes also compute the GGUF reference; they do not
isolate ANE device allocations or system-wide paging. The smaller model file
does **not** establish a runtime RAM reduction.

Output ownership uses at most one physical batch per result slab. The isolated
adapter benchmark for 32 vectors of 768 values at batch size four dropped from
33 to nine Go allocations, with unchanged bytes allocated. Its roughly 46 µs
runtime did not improve materially; the model graph accounts for the measured
end-to-end gains. Retaining one result never retains an entire request slab.

M4 measurements have not been collected.
