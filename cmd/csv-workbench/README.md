# CSV-Arbeitsplatz

Standalone local application with an embedded German browser UI. Runs Laya and
optionally Ministral directly through GopherLLM, without a separate model server.
The compiled binary includes all web assets and works from any directory.

```sh
go build -o /tmp/csv-workbench ./cmd/csv-workbench
/tmp/csv-workbench -model-dir /path/to/models
```

Open http://127.0.0.1:8092. Use `-addr 127.0.0.1:8093` to change the port.
Only loopback addresses are accepted. Startup model flags are optional: choose
and load models in the browser's **Modelle wählen** section. Running without
arguments offers three Laya Hub checkpoints; `-model-dir` adds supported local
GGUF text models and Laya checkpoint directories recursively. Projectors and
embedding-only GGUFs are excluded. The selected GGUF need not be Ministral if
another supported text model is preferred.

For preloaded startup models, use `-laya-model /path/to/checkpoint` and optionally
`-model /path/to/Ministral.gguf`. Hub startup example:
`-laya-model hf:convaiinnovations/laya -laya-subfolder multilingual`.
Explicit startup models are included in the browser catalog even outside the
model directory. The directory is scanned once at startup; restart the app to
include newly added files. Browser **Status aktualisieren** refreshes active
model state, not the filesystem catalog. Browser requests accept catalog IDs,
not arbitrary local paths or repository references.

Use `-hf-offline` for cached Hub models. The first Hub load may download weights;
model selection has a ten-minute timeout and can be cancelled. German input
should use the multilingual Laya checkpoint. Loading failures keep the current
model active. Switching briefly holds both old and new models in memory, so
sufficient memory is required. Selecting the empty entry and pressing the load
button unloads that model. Without Laya, CSV import and existing result export
remain available; classification is disabled. Without a text model, assistant
controls are disabled.

A successful Laya switch clears classifications and corrections, preserving
the imported CSV. Switching the assistant clears its explanations while keeping
Laya results. Model changes and inference share a serialization gate; requests
from stale browser sessions are rejected rather than mixing model versions.
Use **Status aktualisieren** in another open tab after a switch.

1. Select a UTF-8 CSV/TSV file, separator and header mode, then load it. Other
   encodings are rejected rather than replacing characters silently. Limits are
   32 MiB per request and 10,000 records. Original columns and row order survive
   export; quoting and line endings are normalized by Go's CSV reader/writer.
2. Choose an input column or the whole record. Configure the question and a
   category list or label-to-description JSON object. Also supported: yes/no
   probability (`noul`, 0–1) and ordered scale (`score`, zero-based expected level).
   Ministral can propose a question and categories from the task description.
3. Test the first five records. Inspect rows and model probabilities, then
   process all rows. Already classified records are reused. Changing the task,
   criteria, type or input column clears all results to prevent mixing tasks.
4. Filter uncertain, corrected or unprocessed rows. Open a row for its full
   contents and alternative probabilities. Corrections must be valid categories
   or values within the selected scale. Ministral explanations are additional
   interpretations, not Laya's internal reasoning.
5. Export all records. Unprocessed records have empty result fields. Four
   columns are appended: the chosen result name, `<name>_model`,
   `<name>_probability` (maximum option probability, 0–1), and
   `<name>_corrected`. Naming collisions with existing columns are rejected.
   Headerless input produces headerless output. Export uses the imported separator.

Processing is sequential and can be cancelled. Completed rows remain available
for review, export and resumption. A failed row stops processing with an error;
retry continues from the first unprocessed row. Model requests time out after
two minutes. Concurrent inference requests are serialized with cancellation.

Uploads and results are held in browser memory for the current page session.
The server retains no dataset or job; multipart temporary files are removed
after import. Reloading the page clears the workspace. Responses are not cached.
There are no external fonts, scripts or network dependencies in the UI. The
application is intended for one local user; it has no accounts or persistent
storage. Uploaded values are preserved as CSV data, including spreadsheet
formula strings: take care when opening untrusted exports in spreadsheet tools.

Laya has a checkpoint-dependent context budget and can truncate long inputs.
The UI displays this limitation; it does not detect individual truncations.
Model probabilities are not measured accuracy or an authorization signal.

## Checks

```sh
go test ./cmd/csv-workbench
go vet ./cmd/csv-workbench
node --test cmd/csv-workbench/frontend.test.cjs
```

Tests cover CSV quoting, BOM, multiline values, malformed inputs, row limits,
classification request validation, corrected CSV round trips, optional-model
behavior, suggestion validation, request origin/Host checks, cancellation,
model discovery, load rollback, unloading, serialization and stale-model requests.
Frontend tests verify result invalidation, preservation of corrections during
assistant switches, and disabled controls during loading. Automated tests use
fake inference; model accuracy and latency require real checkpoints.
