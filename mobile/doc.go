// Package mobile is GopherLLM's small, stable embedding surface for other
// languages. Every value that crosses it is a string, bool, int or error, and
// options and results travel as JSON, so it maps one-to-one onto the C ABI in
// bindings/c/shim. The Swift package (bindings/swift, for iOS and macOS apps),
// the Rust crate (bindings/rust) and the Python package (bindings/python) are
// all thin layers over that C ABI.
//
// An Engine owns at most one loaded GGUF model:
//
//	e := mobile.NewEngine()
//	defer e.Close()
//	if err := e.Load(path, `{"threads":0}`); err != nil { ... }
//	resultJSON, err := e.Chat(`[{"role":"user","content":"Hi"}]`, `{"max_tokens":64}`)
//
// InspectModel reads a GGUF header without loading weights, so an app can
// show a model's size, architecture and support status before committing
// memory to it.
package mobile
