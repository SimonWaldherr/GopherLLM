# Model Sources

Shows every place GopherLLM can get a model from, and that they all converge on
the same thing: a local `.gguf` path handed to `gopherllm.Open`.

| Source | What it reads | Network |
|---|---|---|
| Model directory | any `.gguf` under `--model-dir` (finds LM Studio's library too) | no |
| Ollama store | models already pulled with `ollama pull`, by `name:tag` | no |
| Hugging Face Hub | `hf:owner/repo[:file.gguf]`, cached locally after the first fetch | yes, on request |

The point of the example is the last few lines of `runModel`: once a reference
resolves to a path, the loading and generation code is identical regardless of
where the model came from.

## Run it

List what is already on the machine — no network, no model required:

```bash
go run ./examples/model-sources
```

Search the Hub:

```bash
go run ./examples/model-sources -search qwen3
```

Generate, from whichever source the reference names:

```bash
go run ./examples/model-sources -run "tinyllama:latest" -prompt "Erklär mir Mmap in einem Satz."
```

```bash
go run ./examples/model-sources -run "hf:bartowski/Qwen2.5-3B-Instruct-GGUF:Qwen2.5-3B-Instruct-Q4_K_M.gguf" -prompt "Hi"
```

## Notes

**Reusing Ollama's models costs nothing.** Ollama stores plain GGUF blobs in a
content-addressed layout, so GopherLLM reads them in place — no copy, no
conversion, no second download. Set `OLLAMA_MODELS` if the store is not at
`~/.ollama/models` (a Linux system service typically uses
`/usr/share/ollama/.ollama/models`).

**Only the Hub touches the network, and only when asked.** An `hf:` reference is
never inferred: a name that matches nothing locally is reported with a
suggestion rather than silently downloaded. Pass `-offline` to use only what is
already cached.

**Ambiguity is surfaced, not guessed.** A repository publishing several
quantizations reports the choice instead of picking one; append `:<file>.gguf`
to select. Use `-search` to find repositories, then
`gopherllm.HuggingFaceRepositoryVariants` (or the CLI's `--hf-list`) to see what
a repository offers.

**Sharded models.** A multi-shard repository resolves to every shard; the
returned path is the first, and the loader finds its siblings. For a checkpoint
too large for RAM, open it with `gopherllm.WithOutOfCore(true)` so the shards
stay memory-mapped instead of being merged into one buffer.
