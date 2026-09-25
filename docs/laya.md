# Native Laya classification

GopherLLM downloads and runs [Laya](https://huggingface.co/convaiinnovations/laya)
decision checkpoints in-process. Inference is Go code using GopherLLM's matrix
kernels, with the existing Accelerate batch path on supported macOS builds.
No Python, PyTorch, ONNX runtime, remote inference service or generated chat
answer is involved. The question describes what to classify in the supplied
state, and the criteria define the possible answers at request time.

## Choose a workflow

| Input | Command / endpoint | Output |
| --- | --- | --- |
| CSV file or Unix pipe | `--classify-csv file.csv` or `--classify-csv -` | CSV on stdout, one appended result per record |
| Browser upload | `/classify` on the running Laya server | Downloadable result CSV |
| HTTP file upload | `POST /v1/systemone/csv` | CSV attachment after the full job succeeds |
| One structured request | `--classify request.json` or `POST /v1/systemone` | Typed JSON answers |

[CSV quickstart](#csv-as-a-unix-filter) · [Browser and upload API](#browser-upload-and-csv-api) ·
[Answer types](#questions-and-answers) · [Troubleshooting](#troubleshooting)

## Download and classify

Build the usual CLI, then run the supplied German example:

```sh
go build -o bin/gopherllm ./cmd/gopherllm
bin/gopherllm --laya-model hf:convaiinnovations/laya \
  --laya-subfolder multilingual --classify examples/laya/request.json
```

The first run downloads only the selected checkpoint's five required files
into the normal Hugging Face snapshot cache. Subsequent runs reuse those files.
`HF_HOME`, `HF_HUB_CACHE`, `HF_ENDPOINT`, `HF_TOKEN`, revision selection and
`--hf-offline` work through the existing Hub client. Model loading never runs
code from a repository. A local directory works without network access:

```sh
# Download without loading or executing the model; prints the local directory.
bin/gopherllm --laya-model hf:convaiinnovations/laya \
  --laya-subfolder multilingual --laya-download-only

# The same request from stdin, using only an already populated cache.
bin/gopherllm --laya-model hf:convaiinnovations/laya \
  --laya-subfolder multilingual --hf-offline --classify - \
  < examples/laya/request.json

# Alternatively use the directory printed by --laya-download-only.
bin/gopherllm --laya-model /path/to/checkpoint --classify request.json
```

Use `hf:owner/repository@commit` to pin a revision. Omit `--laya-subfolder` for
the English root checkpoint, or choose `typed-decisions` for that bundled
fine-tune. The standalone multilingual and typed-decisions repositories can
also be used with their root layout. German and other non-English input should
use the multilingual checkpoint. Native GopherLLM selects the checkpoint
explicitly; it does not reproduce Laya's Python language router.

`--threads N` sets CPU parallelism. `--timeout 30s` bounds a one-shot command,
including download and loading. Cancellation is checked between tensor loads,
transformer stages and attention rows; an active matrix kernel finishes first.

## CSV as a Unix filter

Apply one question to every CSV record and append the answer as its last column:

```sh
cat examples/laya/tickets.csv | bin/gopherllm \
  --laya-model hf:convaiinnovations/laya --laya-subfolder multilingual \
  --classify-csv - --csv-column text \
  --instruction 'Welcher Bereich soll diese Anfrage bearbeiten?' \
  --criteria '["Abrechnung", "Technik", "Vertrieb"]' \
  --result-column category > classified.csv
```

For example, an input record `1,"Bitte meine Rechnung korrigieren"` can become
`1,"Bitte meine Rechnung korrigieren",Abrechnung`. This is an illustrative model
answer; the actual classification depends on the checkpoint, question and criteria.

`--classify-csv input.csv` also accepts a file path or named pipe. `-` means
stdin; stdout contains **only CSV**, while diagnostics go to stderr. The model
is loaded once, records are processed in order, and each output record is
flushed immediately. Memory usage depends on the model and largest record, not the total record count.
CSV header keys are encoded once and output row storage is reused across records.
Use a different output filename from the input when redirecting stdout.

The first record is a header by default. `--csv-column text` selects the named
column as the model's state. Without it, the entire record is supplied as a JSON
object whose keys and order come from the headers. Headers must be nonempty and
unique; an already existing result-column name is rejected. The appended header
defaults to `result`. Original column values and order are retained; CSV quoting
and line endings are normalized by Go's CSV reader/writer. Quoted commas,
embedded newlines, CRLF input and a UTF-8 BOM are supported. Blank physical
lines are skipped according to standard CSV parsing; a quoted multiline field
is one logical record, not several classification jobs.

Additional options:

| Flag | Behavior |
| --- | --- |
| `--instruction 'Question?'` | Required question applied to every record |
| `--criteria '["a","b"]'` | Categories; also accepts a label → description JSON object |
| `--decision-type choice` | Category; default when criteria are supplied |
| `--decision-type noul` | Yes/no probability; default when criteria are absent |
| `--decision-type score --criteria '["low","medium","high"]'` | Expected zero-based scale value |
| `--csv-delimiter ';'` | Same separator for input and output; `\t` selects TSV |
| `--csv-no-header` | Every record is data; `--csv-column 2` selects the second column |
| `--csv-result-json` | Full answer, including probabilities, in the final CSV cell |
| `--result-column category` | Name of the appended header |

Without a header or selected input column, state is an array of field values.
The command stops at the first malformed record, prediction error or output
failure and exits nonzero. Records already written to stdout remain there;
error messages identify the failing **data record number**, excluding the header.
`--timeout` and interruption cancel the job. This is a model instruction,
not an executable shell command or arbitrary text-generation operation.

### Browser upload and CSV API

Start the classifier with `--serve 127.0.0.1:8080` as described below, then open
[the CSV upload page](http://127.0.0.1:8080/classify). Select a file, enter the
question and categories, choose an input column if needed, and download the
CSV with the appended result. The page also offers cancellation, TSV/semicolon
separators, headerless input and full-JSON result cells.

The same operation is available as multipart `POST /v1/systemone/csv`:

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/systemone/csv \
  -F 'file=@examples/laya/tickets.csv' \
  -F 'options=<examples/laya/csv-options.json' \
  -o classified.csv
```

The `options` field is one JSON object:

```json
{
  "question": {
    "type": "choice",
    "instructions": "Welcher Bereich soll diese Anfrage bearbeiten?",
    "criteria": ["Abrechnung", "Technik", "Vertrieb"]
  },
  "input_column": "text",
  "result_column": "category",
  "delimiter": ",",
  "no_header": false,
  "result_json": false
}
```

Optional `max_len` and `head_max_len` select the same per-question budgets as
ordinary decision requests. Exactly one `file` and one `options` field are
required, in either order. The HTTP upload is bounded to 64 MiB including
multipart overhead; output is bounded to 128 MiB. Larger jobs can use the CLI
filter, which imposes no whole-file size limit. Per-record model limits still
apply. The server uses temporary files and removes them after the request,
including on errors and cancellation. It never uses the uploaded filename as
a local path.

Inference is still performed one record at a time. The HTTP result is spooled
until the job succeeds so a late error returns an error response instead of a
partial CSV download. A successful response has `Content-Type: text/csv`,
`Content-Disposition: attachment` and `X-Processed-Rows`. CSV validation errors
return 422, invalid multipart/options return 400, size limits return 413, and
the existing inference deadline applies to the entire upload job (default two
minutes; increase `--request-timeout` for longer jobs). Admission limits and
browser-only deployment restrictions also apply.

### Go streaming interface

```go
stats, err := model.ClassifyCSV(ctx, inputReader, outputWriter,
    gopherllm.DecisionCSVOptions{
        Question: gopherllm.DecisionQuestion{
            Type: "noul",
            Instructions: "Does this message ask for a refund?",
        },
        InputColumn: "text",
        ResultColumn: "refund_probability",
    })
```

The caller owns the streams and model. `gopherllm.ClassifyCSV` also accepts a
`DecisionPredictor` interface for other implementations of the typed decision
API. Returned stats count successfully written records and input tokens.

## Questions and answers

A request contains `state` (text, JSON object or array) and a `questions` object.
Each question has `type`, `instructions` and, when needed, `criteria`:

| Type | Criteria | Answer |
| --- | --- | --- |
| `choice` | Label → description object, or list of string labels | `choice` label and `probabilities` |
| `score` | Ordered list of level descriptions | `score`, the expected zero-based level, plus probabilities and legend |
| `noul` | Optional `false`/`true` descriptions | `noul`, the probability that the statement holds |

`noul` also accepts optional `labels: {"false": "...", "true": "..."}`. Both
labels must be distinct, nonempty strings. Structured criterion descriptions
are serialized as JSON. JSON criterion order is retained because changing
option order can affect the model. Go maps use deterministic sorted key order;
use `json.RawMessage` for an explicitly ordered criterion object.

Every answer includes `answer_confidence` (maximum option probability),
`confidence` (normalized entropy for choice/score, maximum probability for
noul), and `action.act_probability`. Temperatures come from the checkpoint,
including option-count buckets, with the reference runtime's [0.5, 5] clamp.
These values retain Laya's semantics; the action head is not an independently
validated safety signal.

The English root defaults to 512 tokens per question; the supplied multilingual
checkpoint defaults to 1024. Optional request fields `max_len` and
`head_max_len` override the state/prompt budgets, up to 8192 total tokens and
the encoder's configured context limit. Input state is truncated at the token
budget. Oversized question prompts are rejected rather than silently dropping
answer markers. Questions execute sequentially on CPU. Usage reports the sum
of input tokens across questions and zero output tokens.

Limits: 64 questions, 100 options per question, 256 KiB serialized state,
256 KiB criteria per question, and 32 KiB instructions. The HTTP body limit is
2 MiB. An empty question object returns empty answers without inference.

## Web API

```sh
bin/gopherllm --laya-model hf:convaiinnovations/laya \
  --laya-subfolder multilingual --serve 127.0.0.1:8080

curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @examples/laya/request.json
```

`POST /v1/systemone` uses the Jev-style typed decision request/response shape.
The optional `model` field must match the preloaded model ID returned by
`GET /v1/models`; omit it when using the single configured classifier. It never
names a filesystem path to load or initiates a download. The server applies
its request deadline, admission limit, observations and cross-site protection.
`--request-timeout` and `--max-connections` configure the existing controls.
Malformed JSON is 400, invalid question/budget input is 422, oversized bodies
are 413 and unknown model IDs are 404. Browser-only deployment cannot execute
this endpoint.

The CLI's Laya mode is dedicated to classification. To host chat and decision
inference together, pass an existing chat runner plus `DecisionModel` and
`DecisionModelID` in `server.HandlerOptions`/`ServeOptions`. The host owns the
Laya model and closes it after shutting down the handler/server.

A persistent configuration may contain:

```json
{
  "version": 1,
  "laya": {"model": "hf:convaiinnovations/laya", "subfolder": "multilingual"}
}
```

Use it with `--config config.json --serve` or `--classify request.json`.

## Go library

```go
ctx := context.Background()
dir, err := huggingface.DownloadLaya(ctx,
    "convaiinnovations/laya", "multilingual", os.Stderr,
    huggingface.DefaultOptions())
if err != nil { return err }
model, err := gopherllm.OpenLaya(ctx, dir)
if err != nil { return err }
defer model.Close()

result, err := model.Predict(ctx, gopherllm.DecisionRequest{
    State: "Mir wurde der Betrag zweimal abgebucht.",
    Questions: map[string]gopherllm.DecisionQuestion{
        "department": {
            Type: "choice",
            Instructions: "Welcher Bereich soll die Anfrage bearbeiten?",
            Criteria: map[string]string{
                "Abrechnung": "Rechnungen, Zahlungen und Erstattungen",
                "Technik": "Softwarefehler und technische Probleme",
            },
        },
    },
})
if err != nil { return err }
fmt.Println(*result.Answers["department"].Choice)
```

Imports are the root `github.com/SimonWaldherr/GopherLLM` package (aliased
`gopherllm`) and its `huggingface` package, plus the standard library packages
used above. Network dependencies remain outside the root inference package.
`Predict` serializes concurrent calls, supports contexts and returns
`ErrInvalidDecision` for caller input errors. `Close` is idempotent and waits
for active inference.

## Supported checkpoints and validation

The loader supports ModernBERT/mmBERT encoders with alternating global/local
bidirectional RoPE, GeGLU feed-forward layers, Laya's pre-norm ReLU transformer
head, type embeddings, option-marker scorer and act/escalate head. It accepts
F32, F16 and BF16 Safetensors weights. Embeddings remain memory-mapped;
projection weights are expanded to F32 for CPU kernels. Allow roughly twice
the half-precision parameter size in RAM, plus activations. This initial path
does not use Metal, GGUF quantization or out-of-core projections.

Laya-family fine-tunes in this same layout work through `OpenLaya` and
`DownloadLaya`, independent of the repository owner. Other Jev-like models
with different architectures or tokenizer pipelines are rejected with an
explicit error; sharing an HTTP protocol does not imply compatible weights.
Expected files:

```text
rl_agent_config.json
encoder/config.json
tokenizer/tokenizer.json
tokenizer/tokenizer_config.json
model.safetensors
```

Automated tests compare a small synthetic checkpoint against outputs produced
by upstream Laya with PyTorch, including biased encoder layers, alternating
attention windows, both decision head layers, all three question types,
single-option choice, temperatures, token counts and action probabilities.
The fixture has random weights and contains no pretrained model data. Regenerate
it with `scripts/generate_laya_fixture.py` in a separate Python test environment;
Python is never a runtime dependency. Server and CLI tests cover input errors,
lifecycle, deployment restrictions and deadlines.

During implementation the English and multilingual public checkpoints were
also run directly, with tokenizer and decision-output comparisons against the
upstream CPU reference. The full multilingual download → native execution
workflow was exercised on a German refund request, including offline cache
reuse. These are correctness checks, not a claim of task accuracy or calibrated
probabilities on a new application domain.

Reference implementations:
[Laya](https://github.com/NandhaKishorM/laya) and
[Transformers ModernBERT](https://github.com/huggingface/transformers/tree/main/src/transformers/models/modernbert).

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `input column ... not found` | Match the header exactly, or omit `--csv-column` to classify the complete record. |
| `result column ... already exists` | Select a new name with `--result-column`. |
| `wrong number of fields` | Check the delimiter and quoting. The error identifies the data record, excluding the header. |
| Empty/truncated output after a CLI error | Check the exit status and stderr. Streaming output contains only the records completed before the error. |
| HTTP 413 | Use the CLI for files larger than the upload/output limits. |
| HTTP timeout | Increase `--request-timeout` when starting the server. The deadline covers the whole job. |
| Unexpected categories | Use the multilingual model for German and describe categories with a JSON label-to-description object. Evaluate accuracy on representative data. |

A Go context is checked between reads and inference stages; it cannot interrupt
an arbitrary blocked `io.Reader` or `io.Writer`. Callers using pipes or network
streams should close them or apply their own I/O deadlines when cancelling.

### Measuring CSV overhead

The benchmark below uses a fixed predictor and 1,000 six-column records. It
measures parsing, state construction and CSV output, excluding model inference:

```sh
go test ./integration/laya -run '^$' -bench BenchmarkCSVWholeRecords -benchmem -count=3
```

Compare results on the same machine/build. This benchmark does not measure Laya
inference throughput or imply a corresponding end-to-end speedup.

Example comparison on Apple M2 Max, darwin/arm64, `CGO_ENABLED=0` (median of
three runs; each operation processes the complete 1,000-record fixture):

| CSV processing | Before | After |
| --- | ---: | ---: |
| Time per operation | 1.82 ms | 1.09 ms |
| Allocations per operation | 46,044 | 27,063 |
| Allocated bytes per operation | 1,269,439 | 820,934 |

The optimization caches serialized header keys and reuses output-row storage.
Input state remains separate from the output buffer; record order, JSON
escaping and immediate flushing are unchanged.

## Inference performance

Laya attention uses the existing SIMD vector-accumulation kernels. Rotary
frequencies are computed once per dimension and each position's rotation is
shared across attention heads. Large feed-forward activations run across the
worker pool using exact GELU. Tensor workspaces are reused across encoder and
decision-head layers within a forward pass, then released; an unusually long
request does not permanently enlarge the model's scratch storage.

Reproduce the native-inference benchmark using downloaded weights:

```sh
GOPHERLLM_LAYA_BENCH_MODEL=/path/to/checkpoint \
  go test ./integration/laya -run '^$' -bench BenchmarkLayaPredict \
  -benchtime=3x -count=3 -benchmem
```

Loading and a warm-up prediction are excluded. The benchmark uses four GopherLLM
workers, one three-category question, and short/long English states. Without
the environment variable it uses the small synthetic test fixture, which is
useful for development but does not represent real-model speed.

Example on Apple M2 Max, darwin/arm64 with Accelerate enabled, English root
checkpoint, median of three runs (three predictions per run):

| Input | Before | After | Allocated bytes per prediction, before → after |
| --- | ---: | ---: | ---: |
| Short | 138 ms | 91 ms | 54.4 MB → 2.2 MB |
| Long | 1,421 ms | 758 ms | 584.1 MB → 28.2 MB |

These are warm inference measurements on one machine, not a throughput guarantee.
Allocated bytes measure Go allocation traffic, not peak RSS or model-weight
memory. Quantization, token budgets and output formats are unchanged. SIMD
floating-point rounding can differ slightly; the PyTorch golden-reference
checks retain their existing tolerance.
