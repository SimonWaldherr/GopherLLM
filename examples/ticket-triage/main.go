// Ticket Triage reads fake/support JSONL records and emits one local
// human-review draft per input line. It demonstrates batch embedding without
// an HTTP API, database, or third-party service.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

type ticket struct {
	ID      string `json:"id"`
	Subject string `json:"subject"`
	Body    string `json:"body"`
}

type triageResult struct {
	Ticket ticket `json:"ticket"`
	Draft  string `json:"draft"`
}

func main() {
	modelPath := flag.String("model", "", "path to local GGUF model (required)")
	inputPath := flag.String("input", "examples/ticket-triage/tickets.jsonl", "input JSONL file")
	flag.Parse()
	if *modelPath == "" {
		log.Fatal("-model /path/to/model.gguf is required")
	}
	input, err := os.Open(*inputPath)
	if err != nil {
		log.Fatalf("open tickets: %v", err)
	}
	defer input.Close()
	model, err := gopherllm.Open(context.Background(), *modelPath, gopherllm.WithLogWriter(os.Stderr))
	if err != nil {
		log.Fatalf("load local model: %v", err)
	}
	defer model.Close()

	scanner := bufio.NewScanner(input)
	// Support tickets can contain logs; the standard 64 KiB scanner token is
	// unnecessarily restrictive for this local batch example.
	scanner.Buffer(make([]byte, 4<<10), 256<<10)
	encoder := json.NewEncoder(os.Stdout)
	line := 0
	for scanner.Scan() {
		line++
		if strings.TrimSpace(scanner.Text()) == "" {
			continue
		}
		var item ticket
		if err := json.Unmarshal(scanner.Bytes(), &item); err != nil {
			log.Printf("skip line %d: %v", line, err)
			continue
		}
		prompt := fmt.Sprintf("Ticket ID: %s\nSubject: %s\nMessage: %s", item.ID, item.Subject, item.Body)
		answer, err := model.Generate(context.Background(), prompt,
			gopherllm.WithSystemPrompt("You triage customer-support tickets for a human agent. Give a compact draft with: category, priority, missing information, safe first steps, and a customer-facing response. Never promise refunds or repairs. For smoke, burning smell, sparks, swollen batteries, or unusual charging heat: mark urgent safety review and do not suggest continued use or charging."),
			gopherllm.WithMaxTokens(300), gopherllm.WithTemperature(0.2))
		if err != nil {
			log.Printf("ticket %s: %v", item.ID, err)
			continue
		}
		if err := encoder.Encode(triageResult{Ticket: item, Draft: strings.TrimSpace(answer.Text)}); err != nil {
			log.Fatalf("write output: %v", err)
		}
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}
