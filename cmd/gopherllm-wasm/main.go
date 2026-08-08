//go:build js

// Command gopherllm-wasm compiles GopherLLM's inference engine to
// WebAssembly for in-browser use. It registers a small set of JS-callable
// functions (see bridge.go) and otherwise does nothing on its own — all
// control comes from the host page via wasm_exec.js.
package main

func main() {
	registerCallbacks()
	// A Go wasm program's JS-visible functions stay callable only while
	// main is still running; returning tears the instance down.
	select {}
}
