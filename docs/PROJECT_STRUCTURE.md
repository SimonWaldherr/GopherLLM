# Project structure

GopherLLM keeps the public Go package at the repository root so existing
applications can continue importing `github.com/SimonWaldherr/GopherLLM`.
Files in the root therefore contain the public API and the tightly coupled
inference runtime; implementation-only code is grouped below `internal/`.

| Location | Responsibility |
|---|---|
| `cmd/gopherllm/` | CLI entry point and command wiring |
| `server/` | HTTP APIs, streaming, remote integrations, and embedded chat UI |
| `agentos/` | Sandboxed command execution used by agent tools |
| `integration/` | Public API boundary, consumer-module, and opt-in local-model tests |
| `internal/iqcodebook/` | Generated IQ quantization lookup tables |
| `internal/mmapfile/` | OS-specific file mapping and page warm-up backend |
| `internal/metal/` | Objective-C++/Metal compute backend |
| `internal/tooling/` | OpenAI-compatible tool-call wire types and helpers |
| `internal/wordpiece/` | WordPiece normalization tables |
| `testdata/` | Small fixtures, external-consumer module, and preserved profiles |
| `docs/` | User-facing documentation and GitHub Pages assets |

## Placement rules

- Keep an exported inference API or code that must access package-private
  `Runner`/weight internals in the module root.
- Put implementation details that can stand on their own under `internal/`.
- Put black-box tests that import the public module under `integration/`.
- Keep model files, build products, and local caches out of the repository.

This division keeps the import path stable while making platform backends,
generated data, protocol helpers, and black-box tests easy to locate.
