//go:build hestiacapi

package main

/*
#define GOPHERLLM_CGO_EXPORTS
#include "hestia.h"
*/
import "C"

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"path/filepath"
	"runtime/cgo"
	"strings"

	"github.com/SimonWaldherr/GopherLLM/cmd/hestia/internal/hestia"
)

type desktopRuntime struct {
	gate chan struct{}
	core *hestia.Runtime
}

//export hestia_open
func hestia_open(config, dataDir, token *C.char, op uintptr, errorOut **C.char) uintptr {
	ctx, err := operationContext(op)
	if err == nil {
		var cfg hestia.Config
		cfg, err = hestia.DecodeConfig(strings.NewReader(C.GoString(config)))
		if err == nil {
			cfg.DataDir = C.GoString(dataDir)
			if !filepath.IsAbs(cfg.DataDir) {
				setError(errors.New("Hestia data directory must be absolute"), errorOut)
				return 0
			}
			// The native host controls its resident speech model and microphone.
			cfg.Speech, cfg.VoxtralModelPath = hestia.SpeechConfig{}, ""
			var core *hestia.Runtime
			core, err = hestia.OpenRuntime(ctx, cfg, C.GoString(token), io.Discard)
			if err == nil {
				setError(nil, errorOut)
				return uintptr(cgo.NewHandle(&desktopRuntime{gate: make(chan struct{}, 1), core: core}))
			}
		}
	}
	setError(err, errorOut)
	return 0
}

//export hestia_close
func hestia_close(handle uintptr) {
	if host, ok := handleValue[*desktopRuntime](handle); ok {
		host.gate <- struct{}{}
		if host.core != nil {
			_ = host.core.Close()
			host.core = nil
		}
		<-host.gate
		deleteHandle(handle)
	}
}

func hestiaCall(handle, op uintptr, errorOut **C.char, run func(context.Context, *hestia.App) (any, error)) *C.char {
	ctx, err := operationContext(op)
	if err != nil {
		return cResult("", err, errorOut)
	}
	host, ok := handleValue[*desktopRuntime](handle)
	if !ok {
		return cResult("", errors.New("invalid Hestia handle"), errorOut)
	}
	select {
	case host.gate <- struct{}{}:
		defer func() { <-host.gate }()
	case <-ctx.Done():
		return cResult("", ctx.Err(), errorOut)
	}
	if err := ctx.Err(); err != nil {
		return cResult("", err, errorOut)
	}
	if host.core == nil {
		return cResult("", errors.New("Hestia is closed"), errorOut)
	}
	result, err := run(ctx, host.core.App)
	if err != nil {
		return cResult("", err, errorOut)
	}
	data, err := json.Marshal(result)
	return cResult(string(data), err, errorOut)
}

//export hestia_command
func hestia_command(handle uintptr, conversation, text *C.char, op uintptr, errorOut **C.char) *C.char {
	return hestiaCall(handle, op, errorOut, func(ctx context.Context, app *hestia.App) (any, error) {
		command := strings.TrimSpace(C.GoString(text))
		if command == "" || len(command) > 4096 {
			return nil, errors.New("Hestia command needs 1 to 4096 bytes")
		}
		return app.ProcessText(ctx, C.GoString(conversation), command)
	})
}

//export hestia_confirm
func hestia_confirm(handle uintptr, turn, device *C.char, op uintptr, errorOut **C.char) *C.char {
	return hestiaCall(handle, op, errorOut, func(ctx context.Context, app *hestia.App) (any, error) {
		return app.Confirm(ctx, C.GoString(turn), C.GoString(device))
	})
}

//export hestia_add_document
func hestia_add_document(handle uintptr, title, text *C.char, op uintptr, errorOut **C.char) *C.char {
	return hestiaCall(handle, op, errorOut, func(_ context.Context, app *hestia.App) (any, error) {
		if app.Knowledge == nil {
			return nil, errors.New("configure an embedding model first")
		}
		content := C.GoString(text)
		if strings.TrimSpace(content) == "" || len(content) > 1<<20 {
			return nil, errors.New("document must be nonempty and at most 1 MiB")
		}
		result, err := app.Knowledge.AddDocument(C.GoString(title), content)
		if err != nil {
			return nil, err
		}
		return result, nil
	})
}
