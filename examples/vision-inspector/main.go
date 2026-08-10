// Vision Inspector is the smallest useful example of local multimodal
// inference: one GGUF text model, one Pixtral-compatible mmproj, one image.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func main() {
	modelPath := flag.String("model", "", "path to local text GGUF (required)")
	projectorPath := flag.String("mmproj", "", "path to matching Pixtral vision projector GGUF (required)")
	imagePath := flag.String("image", "", "path to local JPEG/PNG/WebP image (required)")
	prompt := flag.String("prompt", "Describe only clearly visible evidence in this image.", "question for the image")
	flag.Parse()
	if *modelPath == "" || *projectorPath == "" || *imagePath == "" {
		log.Fatal("-model, -mmproj, and -image are required")
	}
	image, err := os.ReadFile(*imagePath)
	if err != nil {
		log.Fatalf("read image: %v", err)
	}
	model, err := gopherllm.Open(context.Background(), *modelPath,
		gopherllm.WithVisionProjector(*projectorPath),
		gopherllm.WithLogWriter(os.Stderr),
	)
	if err != nil {
		log.Fatalf("load local vision model: %v", err)
	}
	defer model.Close()
	result, err := model.Chat(context.Background(), []gopherllm.ChatMessage{
		gopherllm.UserMessageWithImages(*prompt, gopherllm.ImageContent{Bytes: image}),
	}, gopherllm.WithSystemPrompt("Answer in the user's language. Describe visible evidence only; say clearly when an image detail is unclear. Do not infer private, medical, or safety-critical facts."), gopherllm.WithMaxTokens(180), gopherllm.WithTemperature(0.2))
	if err != nil {
		log.Fatalf("inspect image: %v", err)
	}
	fmt.Println(strings.TrimSpace(result.Text))
}
