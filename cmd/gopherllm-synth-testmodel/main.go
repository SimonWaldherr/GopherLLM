// Command gopherllm-synth-testmodel writes internal/testmodel's minimal,
// deterministic GGUF so the non-Go binding test suites (bindings/rust,
// bindings/python, bindings/swift) have a real model to run against without
// a binary .gguf checked into the repository:
//
//	go run ./cmd/gopherllm-synth-testmodel -out /tmp/tiny.gguf
//
// It is pure Go (no cgo), so it builds and runs everywhere go build ./...
// does, independent of the C-ABI bindings it exists to test.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/SimonWaldherr/GopherLLM/internal/testmodel"
)

func main() {
	out := flag.String("out", "", "output .gguf path (required)")
	flag.Parse()
	if *out == "" {
		fmt.Fprintln(os.Stderr, "usage: gopherllm-synth-testmodel -out <path.gguf>")
		os.Exit(2)
	}
	if err := testmodel.WriteFile(*out); err != nil {
		fmt.Fprintln(os.Stderr, "gopherllm-synth-testmodel:", err)
		os.Exit(1)
	}
}
