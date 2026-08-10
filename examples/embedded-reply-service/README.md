# Embedded Reply Service — application-owned HTTP surface

This example is for applications that want their own small API rather than the
bundled GopherLLM server. It loads a model directly with `gopherllm.Open` and
serves only two application routes:

```sh
go run ./examples/embedded-reply-service -model /path/to/model.gguf
curl http://127.0.0.1:8091/healthz
curl -X POST http://127.0.0.1:8091/reply \
  -H 'content-type: application/json' \
  -d '{"message":"Explain GGUF in one sentence."}'
```

It intentionally binds to loopback by default and has no authentication,
database, streaming, queue, rate limit, or production error model. Add those
controls before exposing an equivalent service beyond a trusted local machine.
