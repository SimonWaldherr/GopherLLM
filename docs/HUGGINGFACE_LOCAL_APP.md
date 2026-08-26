# Hugging Face Local App integration

This document contains the proposed `huggingface.js` entry for GopherLLM.
The entry is intentionally snippet-based: GopherLLM has no custom URL scheme,
but it accepts Hugging Face GGUF repositories directly and exposes an
OpenAI-compatible local server.

## Ready-to-paste `local-apps.ts` addition

Add the helper next to the other local-app snippets in
`packages/tasks/src/local-apps.ts`:

```ts
const snippetGopherLLM = (model: ModelData, filepath?: string): LocalAppSnippet[] => {
    const modelRef = `hf:${model.id}${getQuantTag(filepath)}`;

    return [
        {
            title: "Install with Go and run GopherLLM",
            setup: [
                "# Requires Go 1.25 or newer:",
                "go install github.com/SimonWaldherr/GopherLLM/cmd/gopherllm@latest",
            ].join("\\n"),
            content: [
                [
                    "# Interactive terminal chat:",
                    `gopherllm ${modelRef} --repl`,
                ].join("\\n"),
                [
                    "# Start an OpenAI-compatible server with the browser UI:",
                    `gopherllm ${modelRef} --serve 127.0.0.1:8080 --chat`,
                ].join("\\n"),
            ],
        },
    ];
};
```

Then add this member to `LOCAL_APPS`:

```ts
gopherllm: {
    prettyLabel: "GopherLLM",
    docsUrl: "https://github.com/SimonWaldherr/GopherLLM/blob/main/docs/HUGGINGFACE_LOCAL_APP.md",
    mainTask: "text-generation",
    displayOnModelPage: isLlamaCppGgufModel,
    snippet: snippetGopherLLM,
},
```

`getQuantTag(filepath)` is the helper already used by the existing GGUF apps.
It produces a concrete quantization when the model page is showing a specific
GGUF file and preserves Hugging Face's `{{QUANT_TAG}}` placeholder when the
repository has multiple selectable quantizations.

## Why this matches GopherLLM

- The CLI accepts `hf:owner/repository[:quant][@revision]` as its positional
  model selector.
- GGUF shards are downloaded and reused through the Hugging Face cache.
- `--repl` starts an interactive local session.
- `--serve 127.0.0.1:8080 --chat` starts the local HTTP API and the `/chat`
  browser UI.
- The server includes `/v1/chat/completions`, so existing OpenAI-compatible
  clients can connect to `http://127.0.0.1:8080/v1`.

The entry deliberately uses `isLlamaCppGgufModel`, matching the other GGUF
local apps. Model support remains governed by GopherLLM's GGUF architecture
support and its normal load-time validation.

## Local verification

The command contract is covered by `cmd/gopherllm/main_test.go`. Run:

```sh
go test ./cmd/gopherllm
go build ./cmd/gopherllm
```

With a real public GGUF repository, the two generated commands can then be
smoke-tested as follows:

```sh
gopherllm hf:bartowski/Qwen3-4B-GGUF:Q4_K_M --repl
gopherllm hf:bartowski/Qwen3-4B-GGUF:Q4_K_M --serve 127.0.0.1:8080 --chat
```
