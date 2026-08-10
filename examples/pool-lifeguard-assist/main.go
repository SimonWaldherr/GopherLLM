// Pool Lifeguard Assist (CLI) is the smallest local, non-UI companion to the
// example described in README.md. It sends one or more already-captured
// frames from a labelled zone to a local GopherLLM vision model and asks for
// a cautious, evidence-only description an operator can act on.
//
// This is an operator-assistance proof of concept, not a drowning-detection
// system. It does not decide that an emergency exists and it does not trigger
// any alarm, notification, or rescue action. See README.md for the safe
// operating pattern this CLI is meant to be used within.
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
	zone := flag.String("zone", "Pool A", "operator-owned zone or camera label")
	frames := flag.String("frames", "", "comma-separated paths to one or more local JPEG/PNG/WebP frames (required)")
	flag.Parse()
	if *modelPath == "" || *projectorPath == "" || *frames == "" {
		log.Fatal("-model, -mmproj, and -frames are required")
	}

	var images []gopherllm.ImageContent
	for _, path := range strings.Split(*frames, ",") {
		path = strings.TrimSpace(path)
		if path == "" {
			continue
		}
		data, err := os.ReadFile(path)
		if err != nil {
			log.Fatalf("read frame %q: %v", path, err)
		}
		images = append(images, gopherllm.ImageContent{Bytes: data})
	}
	if len(images) == 0 {
		log.Fatal("-frames did not contain any usable image path")
	}

	model, err := gopherllm.Open(context.Background(), *modelPath,
		gopherllm.WithVisionProjector(*projectorPath),
		gopherllm.WithLogWriter(os.Stderr),
	)
	if err != nil {
		log.Fatalf("load local vision model: %v", err)
	}
	defer model.Close()

	prompt := fmt.Sprintf(
		"Watch %s for smoke, crowding, a person in visible distress, or an obstructed view. "+
			"State only what is visible in one sentence. If unclear, say so and ask the operator to check the zone.",
		*zone,
	)
	result, err := model.Chat(context.Background(), []gopherllm.ChatMessage{
		gopherllm.UserMessageWithImages(prompt, images...),
	},
		gopherllm.WithSystemPrompt("You are a secondary pair of eyes for an already-supervised pool or spa, not a drowning-detection or emergency-decision system. "+
			"Describe only clearly visible evidence in the supplied frame(s) for the named zone. Say clearly when the view is unclear, obstructed, or insufficient. "+
			"Never state that a person is drowning or in medical distress; instead describe what is visible and recommend the operator check the zone in person. "+
			"Do not recommend contacting emergency services, sounding alarms, or any action beyond a human operator check."),
		gopherllm.WithMaxTokens(180),
		gopherllm.WithTemperature(0.2),
	)
	if err != nil {
		log.Fatalf("describe zone: %v", err)
	}
	fmt.Println(strings.TrimSpace(result.Text))
}
