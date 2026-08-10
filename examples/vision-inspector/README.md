# Vision Inspector — direct local package example

Runs one image through a local Pixtral-compatible GopherLLM vision model. It
does not start an HTTP server or upload the image anywhere.

```sh
go run ./examples/vision-inspector \
  -model /path/to/text-model.gguf \
  -mmproj /path/to/mmproj.gguf \
  -image ./photo.jpg \
  -prompt "Read the clearly visible text and name the main object."
```

The important integration boundary is small:

```go
model, _ := gopherllm.Open(ctx, textGGUF, gopherllm.WithVisionProjector(mmprojGGUF))
result, _ := model.Chat(ctx, []gopherllm.ChatMessage{
    gopherllm.UserMessageWithImages(prompt, gopherllm.ImageContent{Bytes: image}),
})
```

Use a projector that actually matches the text model. GopherLLM supports its
validated Pixtral vision path; an arbitrary `mmproj` file is not necessarily
compatible. This example is for local image inspection, not identity,
biometric, medical, or safety-critical decisions.
