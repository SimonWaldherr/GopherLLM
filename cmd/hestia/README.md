# Hestia demo

Hestia is a local assistant with its own HTTP UI and simulated lamps. The
default configuration runs the text-command path without loading a model.
Optional Voxtral and embedding checkpoints enable speech and document search.

This directory is a separate Go module requiring Go 1.27.1 or later. It uses
the parent GopherLLM checkout and a `tinyRAG` checkout next to GopherLLM:

```text
workspace/
  GopherLLM/
    cmd/hestia/
  tinyRAG/
```

The relative replacements in `go.mod` express this local development layout.
The first build downloads this module's existing third-party dependencies.

Run and test from this directory:

```sh
cd cmd/hestia
go test ./...
go run .
```

Open [http://127.0.0.1:8787](http://127.0.0.1:8787). Ask
`Schalte die Stehlampe im Wohnzimmer ein.` to operate the simulated floor
lamp. Ambiguous requests ask for clarification. No physical devices are
connected by this demo.

By default, the process stores its journal in `./hestia-data`. To change the
address or data location, pass `-config /path/to/config.json` with, for example:

```json
{"listen":"127.0.0.1:8787","data_dir":"/tmp/hestia-demo-data"}
```

Use `-voxtral-model /path/to/voxtral.gguf` or
`-embed-model /path/to/embedding.gguf` to load the optional models. Without
them, voice and document search remain disabled. Keep the demo on loopback;
it has no authentication.

The repository root's `go test ./...` does not enter nested Go modules, so
Hestia needs the explicit test command above. Its tests use simulated devices
and synthetic model fixtures; they do not measure real-model quality.
