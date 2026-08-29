// Tool Calling is the smallest complete tool-calling program: one Go
// function becomes a model tool via gopherllm.NewTool, and the model
// executes it and answers using the result — no HTTP server, no manual
// tool_calls round-trip.
//
// The tool is real arithmetic, not a canned string, specifically so the
// output is self-verifying: a small local model asked to compute
// "1234 * 5678 + 42" directly will often get it wrong, while the same
// question routed through the calculate tool below is exactly right, because
// the model never actually does the arithmetic — it reads the expression off
// to the tool and reports back what came back.
//
//	go run ./examples/tool-calling -model /path/to/model.gguf
//	go run ./examples/tool-calling -model /path/to/model.gguf -prompt "What is (12 + 8) * 3?"
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// CalculateArgs' tags are all NewTool needs to derive a JSON Schema by
// reflection: `desc` becomes the property description a small model actually
// reads to decide how to call the tool, and the field is required because it
// carries neither `omitempty` nor a pointer type.
type CalculateArgs struct {
	Expression string `json:"expression" desc:"An arithmetic expression using +, -, *, /, and parentheses, e.g. \"(3 + 4) * 2\""`
}

func main() {
	modelPath := flag.String("model", "", "path to local GGUF model (required)")
	prompt := flag.String("prompt", "What is 1234 * 5678, plus 42?", "question to ask the model")
	flag.Parse()
	if *modelPath == "" {
		log.Fatal("-model /path/to/model.gguf is required")
	}

	ctx := context.Background()
	model, err := gopherllm.Open(ctx, *modelPath, gopherllm.WithLogWriter(os.Stderr))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	calculate := gopherllm.NewTool("calculate",
		"Evaluate an exact arithmetic expression. Always use this for a calculation instead of computing it yourself.",
		func(_ context.Context, args CalculateArgs) (string, error) {
			result, err := evaluate(args.Expression)
			if err != nil {
				return "", err
			}
			return formatResult(result), nil
		})
	calculate.Trusted = true // deterministic local arithmetic, not external or attacker-influenced data

	// RunAgenticChatWithTools/RunAgenticChatObserved predate the GenOption
	// veneer Model.Chat/Generate/Stream use and take a plain GenerationOptions;
	// ApplyGenOptions builds one the same way, ctx included.
	options := gopherllm.ApplyGenOptions(ctx, gopherllm.WithTemperature(0), gopherllm.WithMaxTokens(200))

	result, err := gopherllm.RunAgenticChatObserved(model.Runner(),
		[]gopherllm.ChatMessage{gopherllm.UserMessage(*prompt)},
		options, nil, []gopherllm.AgenticTool{calculate}, nil,
		printToolActivity)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Text)
}

// formatResult renders result in plain decimal, never scientific notation:
// %g switches to an "e+06"-style exponent once a number gets a handful of
// digits (7006694 included), which reads fine in Go source but is a
// needlessly confusing thing for a model — or a person watching stderr — to
// have to reparse back into the number it actually is.
func formatResult(result float64) string {
	return strconv.FormatFloat(result, 'f', -1, 64)
}

// printToolActivity narrates each tool call to stderr, so stdout stays just
// the model's final answer (pipeable) while the calculation the model relied
// on is still visible for a reader running this the first time.
func printToolActivity(e gopherllm.AgentEvent) {
	switch e.Kind {
	case gopherllm.AgentEventToolStart:
		fmt.Fprintf(os.Stderr, "[tool] %s(%s)\n", e.Tool, e.Arguments)
	case gopherllm.AgentEventToolEnd:
		if e.Error != "" {
			fmt.Fprintf(os.Stderr, "[tool] %s failed: %s\n", e.Tool, e.Error)
		} else {
			fmt.Fprintf(os.Stderr, "[tool] %s -> %s\n", e.Tool, e.Result)
		}
	}
}
