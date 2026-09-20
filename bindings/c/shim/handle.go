//go:build capi

package main

import (
	"runtime/cgo"

	"github.com/SimonWaldherr/GopherLLM/mobile"
)

// Engine handles cross the C ABI as a plain uintptr (cgo maps Go's uintptr to
// C's uintptr_t in generated exported signatures). cgo.Handle already gives a
// safe, GC-cooperative way to turn a Go pointer into an opaque integer a
// foreign caller can hold and hand back later; we only need a thin wrapper
// that reports a lookup failure as a bool instead of panicking on a bad or
// already-freed handle; a foreign caller passing a stale handle is a
// programming error we want to report through the same "return an error
// string" convention every other shim function uses, not crash the process.
func newEngineHandle() uintptr {
	return uintptr(cgo.NewHandle(mobile.NewEngine()))
}

func engineFromHandle(h uintptr) (engine *mobile.Engine, ok bool) {
	if h == 0 {
		return nil, false
	}
	defer func() {
		if recover() != nil {
			engine, ok = nil, false
		}
	}()
	v := cgo.Handle(h).Value()
	engine, ok = v.(*mobile.Engine)
	return engine, ok
}

func deleteEngineHandle(h uintptr) {
	if h == 0 {
		return
	}
	defer func() { recover() }()
	cgo.Handle(h).Delete()
}
