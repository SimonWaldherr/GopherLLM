//go:build js

// Package webgpu is a thin, hand-written Go wrapper over the browser's
// native WebGPU API (accessed via syscall/js), mirroring how internal/metal
// wraps Apple's Metal API for the darwin/cgo build. No third-party
// JS/WASM code is used anywhere in this package — only the browser's own
// navigator.gpu surface.
package webgpu

import "syscall/js"

// await blocks the calling goroutine until the JS Promise p settles,
// returning its resolved value or an error built from its rejection
// reason. Safe to call from any goroutine: Go's js/wasm scheduler yields to
// the browser event loop cooperatively while a goroutine is blocked on a
// channel receive, so this never stalls the page.
func await(p js.Value) (js.Value, error) {
	type result struct {
		value js.Value
		err   error
	}
	ch := make(chan result, 1)

	var onResolve, onReject js.Func
	onResolve = js.FuncOf(func(this js.Value, args []js.Value) any {
		var v js.Value
		if len(args) > 0 {
			v = args[0]
		}
		ch <- result{value: v}
		return nil
	})
	onReject = js.FuncOf(func(this js.Value, args []js.Value) any {
		reason := "rejected"
		if len(args) > 0 {
			reason = rejectionMessage(args[0])
		}
		ch <- result{err: errString(reason)}
		return nil
	})
	// Released once this call's single settlement has been observed — each
	// await call creates its own pair, so holding them open past that first
	// (and only) invocation would just leak; this matters on a per-token
	// hot path where device/buffer operations recur every generation step.
	defer onResolve.Release()
	defer onReject.Release()

	p.Call("then", onResolve, onReject)
	r := <-ch
	return r.value, r.err
}

type errString string

func (e errString) Error() string { return string(e) }

// rejectionMessage extracts a human-readable string from a JS rejection
// value without calling js.Value.String() on a value that might not be a
// JS string (only well-defined for TypeString).
func rejectionMessage(v js.Value) string {
	switch v.Type() {
	case js.TypeString:
		return v.String()
	case js.TypeObject:
		if msg := v.Get("message"); msg.Type() == js.TypeString {
			return msg.String()
		}
	}
	return js.Global().Get("String").Invoke(v).String()
}
