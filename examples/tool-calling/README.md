# Tool Calling — the smallest complete tool-calling program

Loads a local GGUF model directly from Go and gives it one tool: a real
arithmetic evaluator, exposed via `gopherllm.NewTool` with no schema string to
keep in sync (the Go struct's tags are the whole schema). No HTTP server, no
manual `tool_calls` round-trip, no fake data.

```sh
go run ./examples/tool-calling -model /path/to/model.gguf
go run ./examples/tool-calling -model /path/to/model.gguf -prompt "What is (12 + 8) * 3?"
```

The tool call and its result print to stderr as they happen; only the model's
final answer goes to stdout, so `... > answer.txt` captures just that.

The tool is genuine arithmetic rather than a canned string on purpose: ask a
small local model to compute `1234 * 5678 + 42` directly and it will often
get the exact value wrong, while the same question routed through
`calculate` is exactly right, because the model never does the arithmetic
itself — it reads the expression to the tool and reports what came back. The
evaluator ([`evaluate.go`](evaluate.go)) reuses `go/parser` for correct
operator precedence and parentheses; it only ever descends into numeric
literals and `+ - * /`, so a malformed or unexpected expression is rejected
rather than evaluated as Go source.

Tool-calling reliability is the model's, not GopherLLM's: this was tested
against Qwen2.5-Instruct at 1.5B and 3B (both `bartowski/Qwen2.5-*-Instruct-GGUF`
on Hugging Face, `Q4_K_M`). At 1.5B the model sometimes skips the tool and does
the arithmetic itself (getting it wrong), or emits a slightly malformed
call tag; at 3B it reliably calls `calculate` and answers correctly for the
default prompt and simple variations. If tool calls aren't happening or the
model emits something that isn't quite `<tool_call>{...}</tool_call>`, that
is the model failing to follow its own rendered instructions, not a parser
GopherLLM needs to be taught a new format — try a larger or more
instruction-tuned model before assuming the framework is at fault.

See [`main.go`](main.go) for the whole program — it fits on one screen. The
[root README's "Tools with a Go function"](../../README.md#tools-with-a-go-function)
section walks through the same pattern; the [`agent` package](../../README.md#retrieval-over-your-own-documents)
is the batteries-included alternative once a tool needs to search your own
documents or return citations.
