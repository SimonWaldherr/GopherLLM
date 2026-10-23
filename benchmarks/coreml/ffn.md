# Core ML prefill FFNs

`scripts/export_coreml_ffn.py` exports one selected dense Qwen or Ministral
SwiGLU block. The fixed interface is `hidden[T,D] -> output[T,D]`, using FP32
I/O and FP16 internal projections. Attention, norms, residuals and KV/recurrent
state stay in GopherLLM. This is an optional approximate prefill path, not a
conversion of the full chat model to Core ML.

## Export and attach

Use the optional Python environment described in [the encoder benchmark](README.md).
Keep exports at persistent paths so Core ML can reuse specialization caches.

```sh
python scripts/export_coreml_ffn.py model.gguf ffn-l0.mlpackage \
  --layer 0 --tokens 32 --compile-to ffn-l0.mlmodelc
```

For the measured Qwen3.8-27B layer-0 CPU variant, add
`--activation-scale 8 --hidden-chunk 4352` and attach it with
`CoreMLFFNOptions{CPUOnly: true}`. The measured Ministral-3-3B variant uses the
default scale 16, an unsplit hidden dimension and default CPU/ANE permission.

The exporter writes `ffn-l0.json` beside the compiled model. It identifies the
source architecture, layer, dimensions, graph options and SHA-256 of each of the
three original encoded GGUF tensors. The same source binding is embedded in the
model. Keep the manifest and its matching compiled model together.

```go
model, err := gopherllm.Open(ctx, "model.gguf")
if err != nil { return err }
defer model.Close()
err = model.Runner().SetCoreMLFFN([]string{"ffn-l0.json"}, gopherllm.CoreMLFFNOptions{})
if err != nil { return err }
```

Build with `CGO_ENABLED=1` and `-tags 'metal coreml'` on macOS 13/iOS 16 or
newer. Without `coreml`, attaching a nonempty selection returns
`coreml.ErrUnavailable`. Passing `nil` removes the selection and releases the
native models. `CoreMLFFNOptions{CPUOnly: true}` selects CPU-only execution for
comparison; the default permits CPU and Neural Engine execution. Permission is
not a measurement of ANE utilization.

Loading checks both source hashes and compiled metadata. It also compares three
differently scaled deterministic rows against source-weight FP32 reference dots.
Each row must have cosine >=0.9999 and relative L2 <=1%. Rejection leaves the
previous selection intact. This smoke check does not replace generation and
corpus quality evaluation. At prediction time, a non-finite FP16 result retries
the complete stateless block on its original backend; other native errors are
returned to the caller.

Only complete fixed token blocks are offloaded. Smaller or uneven chunks retain
the original FFN, and a complete whole-chunk Metal prefill takes precedence.
Hybrid Qwen's chunks have at most 32 rows; larger exports are rejected there.
Ordinary decode, speculative verification and Qwen's MTP-prefill path retain
their existing arithmetic. Replacing the selection invalidates conversational
and prefix-cache states computed with the previous arithmetic.

The original GGUF weights remain available for all these paths. Core ML adds
weight storage; this feature does not reduce total model RAM. Start with one
layer and measure end-to-end latency before adding more. A faster isolated FFN
does not prove a faster full prefill, especially when it introduces additional
CPU/GPU synchronization.

## Reproduce a layer comparison

```sh
GOPHERLLM_COREML_FFN_GGUF=/path/to/model.gguf \
GOPHERLLM_COREML_FFN_PATH=/path/to/ffn-l0.mlmodelc \
GOPHERLLM_COREML_FFN_UNITS=cpu,ane \
  CGO_ENABLED=1 go test -tags 'metal coreml' . \
  -run '^TestCoreMLConfiguredFFN$' -count=1 -v
```

The benchmark maps the GGUF and loads only the gate/up/down matrices. It avoids
loading the vocabulary, attention layers or a full Runner. JSON records compare
the ordinary CPU and custom fused Metal paths with Core ML. Model load and first
prediction are separated from individual-request median/P90 over nine windows
of at least 50 ms after warmup. Timings include the host/device round trip.
`UNITS` also accepts `gpu` and `all` for diagnostics. The optional `MANIFEST`
variable uses the same prefix; otherwise the JSON path is derived from the
compiled model name.

All paths report error against a source-weight reference with unquantized
activations. The ordinary CPU path may use its existing Q8-activation kernels,
so its error is reported too. Numerical results are report-only in this
exploratory harness; native prediction errors, invalid source bindings and
non-finite values fail. `quality_probe_accepted` separately reports whether
the runtime would accept that model/device combination. Do not treat a timing
from a rejected variant as a usable speedup.

## Numerical changes

Direct Core ML SiLU/Swish and FP16 operations on small intermediate values gave
large errors for real Ministral FFNs on the M2 Max. The exporter uses an explicit
stable SiLU representation with `exp(-abs(x))` rather than Swish. Projection
scaling and final FP32 compensation protect small intermediates; all scaling
parameters are part of the compiled source binding. Final FP32 compensation
must stay excluded from Core ML's FP16 conversion pass.

`--activation-scale` is a power of two, default 16. Gate/up weights are
multiplied by this factor, SiLU uses the original unscaled argument while
retaining the scaled numerator, and the final FP32 output is divided by its
square. It trades underflow protection against FP16 headroom; it is not a
universal optimum. Qwen3.8-27B layer 0 required scale 8 in this workload:
scale 16 passed the timing input but overflowed on a different quality-probe
row and was rejected. The exporter rejects non-finite scaled weights before
conversion, and the runtime checks actual predictions too.

`--hidden-chunk` optionally splits the hidden dimension into multiple complete
gate/up/down branches whose down-projections are summed. This can avoid a very
wide hidden axis, but changes FP16 reduction order and is not assumed faster.
`--weight-compression palette8` is optional, applies only to convolution weights,
and must pass the same source-reference checks.

Native exporter regression tests, including large positive/negative activation
inputs, are opt-in:

```sh
PYTHONDONTWRITEBYTECODE=1 GOPHERLLM_COREML_EXPORT_TESTS=1 \
  python -m unittest scripts/test_export_coreml_ffn.py
```

## M2 Max results, 2026-10-24

Host: M2 Max, 32 GiB, macOS 27.2, Go 1.25.3, coremltools 9.0. Other
applications remained active. Both checkpoints are Q4_K_M GGUFs. Each paired
comparison used the same layer-0 weights and 32 deterministic hidden-state
rows; CPU/ANE execution order was reversed in a separate run. This includes
the final exported graph and runtime quality probe.

| Model / export / permitted devices | Custom Metal median | Core ML median | Core ML P90 |
|---|---:|---:|---:|
| Ministral-3-3B / scale 16 / CPU + ANE | 3.584–3.593 ms | **3.260–3.276 ms** | 3.300–3.321 ms |
| Qwen3.8-27B / scale 8, hidden chunks 4352 / CPU only | 9.128–9.139 ms | **5.787–5.795 ms** | 5.860–5.883 ms |

These are approximately 8.8–9.0% and 36.5–36.7% lower isolated FFN latency,
respectively. Qwen's improvement uses Core ML on the CPU, not the Neural
Engine. Its CPU/ANE mode was correct but took 9.203–9.285 ms, with no advantage
over custom Metal. Core ML GPU variants were also correct but slower than
custom Metal here.

Relative L2 error against the float-activation source reference was 0.06954%
for Ministral and 0.54162% for Qwen's CPU variant. Minimum row cosines were
0.999999736 and 0.999986526. Both selected combinations passed the runtime's
separate differently scaled probe in both runs. Qwen scale 16 overflowed on
that probe; CPU-only Qwen without the four hidden branches also failed
numerical checks. These variants are not recommended despite their timings.

[Raw measurements](m2-max-ffn-2026-10-24.csv) retain exploratory and rejected
variants as well as final runs. Missing probe results in older exploratory runs
are unknown, not passes. An unsplit FP16 layer export adds approximately
170 MB of weight storage for Ministral and 535 MB for Qwen before Core ML's
own native allocations. Splitting does not reduce the total weight count.
RAM, paging, M4 and Ministral-14B speedups have not been established.

A separate complete Ministral CPU-prefill check loaded the Runner, processed a
32-token German/English text prefix and replaced only layer 0. Against the same
float-activation reference, final-logit cosine was 0.9999999924 and relative L2
0.01708%. Its single warm timings (1084 vs 1070 ms) are not a statistically
established full-model speedup. Reproduce it separately:

```sh
GOPHERLLM_COREML_PREFILL_GGUF=/path/to/model.gguf \
GOPHERLLM_COREML_PREFILL_MANIFEST=/path/to/ffn-l0.json \
  CGO_ENABLED=1 go test -tags 'metal coreml' . \
  -run '^TestCoreMLConfiguredPrefill$' -count=1 -v
```

An earlier synthetic-token comparison against the ordinary Q8-activation CPU
path showed much larger logit drift (cosine 0.98452, relative L2 17.63%). That
path uses a different activation approximation; these results do not establish
quality equivalence to existing Q8 CPU generation. Evaluate actual prompts and
outputs before deployment. No automatic backend switch was enabled.
