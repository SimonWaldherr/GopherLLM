# Parakeet FFN activation scheduling

Measured on 2026-10-25 with an Apple M2 Max, 32 GiB RAM, Go 1.25.3 and
Darwin Accelerate. Both binaries use eight Go workers. The reference includes
the preceding speech-frontend improvements and differs only in
`parakeetFeedForwardBatch`.

The FFN distributes independent Swish rows over the existing worker pool for
batches of at least 16 frames with a hidden width of at least 512. Each value
uses the unchanged activation formula. Short/narrow batches stay inline.
Returning the existing projected row views after the residual update removes
an otherwise redundant row-header array.

Four samples per version ran in ABBA ABBA order, with a 500 ms benchmark
target and no concurrent builds from this task. The existing benchmarks use
44 frames, model width 1024, FFN width 4096 and synthetic F32 weights at the
Parakeet-TDT dimensions. Setup and weight construction are outside timing.

| Workload | Before median | After median | Time reduction |
| --- | ---: | ---: | ---: |
| FFN half-step including normalization, projections and residual | 2.226 ms | 1.420 ms | 36.2% |
| Both FFNs, attention and convolution | 9.483 ms | 7.823 ms | 17.5% |
| Unchanged attention, control | 2.096 ms | 2.096 ms | approximately zero |

`BenchmarkParakeetEncoderLayer` includes the two FFNs, relative attention and
convolution but excludes the encoder block's final LayerNorm. These are
component measurements, not full-model transcription throughput.

Allocations remain seven per FFN and 45 for the combined stages. Allocation
volume falls by about 1.1 KiB per FFN because the removed 44-row header array
outweighs the new dispatch closure. No extra activation or weight matrix is
allocated. [All samples](m2-max-parakeet-ffn-workers-2026-10-25.csv).

The regression test compares every output bit with the original batched graph,
using one, three and eight workers, empty/short/uneven batches and widths below
and above the dispatch threshold. The existing per-frame reference also checks
that inputs are not modified. Native, portable, scalar-SIMD opt-out and targeted
race tests passed.

To reproduce, compile the reference and changed source separately, then
alternate the resulting binaries in the order above:

```sh
CGO_ENABLED=1 go test -c -o /tmp/parakeet.test .
GOMAXPROCS=8 /tmp/parakeet.test -test.run '^$' -test.benchtime=500ms \
  -test.bench '^BenchmarkParakeet(EncoderLayer|FeedForwardBatch|RelPosSelfAttention)$'
go test . -run '^TestParakeetFeedForward' -count=1
```
