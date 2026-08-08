//go:build js

package main

import "syscall/js"

// newPromise runs run in a new goroutine and returns a JS Promise that
// settles from whatever run does with the resolve/reject functions it's
// handed. This is the bridge between Go's goroutine model and JS's Promise
// model: a goroutine can block on a channel (e.g. while awaiting another
// Promise via awaitPromise, or while a model forward pass runs) without
// stalling the browser's event loop, since Go's js/wasm scheduler yields to
// it cooperatively between goroutine steps.
func newPromise(run func(resolve, reject js.Value)) js.Value {
	executor := js.FuncOf(func(this js.Value, args []js.Value) any {
		resolve, reject := args[0], args[1]
		go run(resolve, reject)
		return nil
	})
	// The executor closure must outlive the Promise constructor call (the
	// JS runtime invokes it synchronously), so releasing it right after
	// New returns is safe.
	defer executor.Release()
	return js.Global().Get("Promise").New(executor)
}

// rejectPromise builds an already-rejected Promise carrying err's message,
// for JS-callable functions that fail argument validation before any
// goroutine work would begin.
func rejectPromise(err error) js.Value {
	return newPromise(func(resolve, reject js.Value) {
		reject.Invoke(err.Error())
	})
}

// awaitPromise blocks the calling goroutine until the JS Promise p settles,
// returning its resolved value or an error built from its rejection reason.
// Safe to call from a goroutine started by newPromise's run function.
func awaitPromise(p js.Value) (js.Value, error) {
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
	defer onResolve.Release()
	defer onReject.Release()

	p.Call("then", onResolve, onReject)
	r := <-ch
	return r.value, r.err
}

type errString string

func (e errString) Error() string { return string(e) }

// rejectionMessage extracts a human-readable string from a JS Promise
// rejection value without ever calling js.Value.String() on a value that
// might not be a JS string (which is only well-defined for TypeString).
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
