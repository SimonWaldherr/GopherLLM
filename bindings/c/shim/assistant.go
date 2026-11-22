//go:build capi

package main

/*
#define GOPHERLLM_CGO_EXPORTS
#include "gopherllm.h"
*/
import "C"

import (
	"context"
	"errors"
	"runtime/cgo"
	"unsafe"

	"github.com/SimonWaldherr/GopherLLM/huggingface"
	"github.com/SimonWaldherr/GopherLLM/mobile"
	"github.com/SimonWaldherr/GopherLLM/speech"
)

type operation struct {
	ctx    context.Context
	cancel context.CancelFunc
}

//export gopherllm_operation_new
func gopherllm_operation_new() uintptr {
	ctx, cancel := context.WithCancel(context.Background())
	return uintptr(cgo.NewHandle(&operation{ctx, cancel}))
}

//export gopherllm_operation_cancel
func gopherllm_operation_cancel(handle uintptr) {
	if op, ok := handleValue[*operation](handle); ok {
		op.cancel()
	}
}

//export gopherllm_operation_free
func gopherllm_operation_free(handle uintptr) {
	if op, ok := handleValue[*operation](handle); ok {
		op.cancel()
		deleteHandle(handle)
	}
}

func operationContext(handle uintptr) (context.Context, error) {
	if handle == 0 {
		return context.Background(), nil
	}
	op, ok := handleValue[*operation](handle)
	if !ok {
		return nil, errors.New("invalid operation handle")
	}
	return op.ctx, op.ctx.Err()
}

//export gopherllm_speech_available
func gopherllm_speech_available(backend *C.char) C.int {
	if speech.Available(speech.Backend(C.GoString(backend))) {
		return 1
	}
	return 0
}

//export gopherllm_speech_open
func gopherllm_speech_open(path, optionsJSON *C.char, op uintptr, errorOut **C.char) uintptr {
	ctx, err := operationContext(op)
	if err != nil {
		setError(err, errorOut)
		return 0
	}
	model, err := mobile.OpenSpeech(ctx, C.GoString(path), C.GoString(optionsJSON))
	setError(err, errorOut)
	if err != nil {
		return 0
	}
	return uintptr(cgo.NewHandle(model))
}

//export gopherllm_speech_free
func gopherllm_speech_free(handle uintptr) {
	if model, ok := handleValue[*speech.Recognizer](handle); ok {
		_ = model.Close()
		deleteHandle(handle)
	}
}

//export gopherllm_speech_transcribe
func gopherllm_speech_transcribe(handle uintptr, samples *C.float, count C.uint32_t, activation *C.char, op uintptr, errorOut **C.char) *C.char {
	ctx, err := operationContext(op)
	if err != nil {
		return cResult("", err, errorOut)
	}
	model, ok := handleValue[*speech.Recognizer](handle)
	if !ok {
		return cResult("", errors.New("invalid speech handle"), errorOut)
	}
	if samples == nil || count == 0 || count > speech.MaxSamples {
		return cResult("", errors.New("speech requires 1 to 480000 float32 samples"), errorOut)
	}
	var phrase *string
	if activation != nil {
		text := C.GoString(activation)
		phrase = &text
	}
	input := unsafe.Slice((*float32)(unsafe.Pointer(samples)), int(count))
	transcript, err := mobile.TranscribeSpeech(ctx, model, input, phrase)
	return cResult(transcript, err, errorOut)
}

//export gopherllm_hub_search
func gopherllm_hub_search(query, optionsJSON *C.char, op uintptr, errorOut **C.char) *C.char {
	ctx, err := operationContext(op)
	if err != nil {
		return cResult("", err, errorOut)
	}
	result, err := mobile.HubSearch(ctx, C.GoString(query), C.GoString(optionsJSON))
	return cResult(result, err, errorOut)
}

//export gopherllm_hub_inspect
func gopherllm_hub_inspect(reference, optionsJSON *C.char, op uintptr, errorOut **C.char) *C.char {
	ctx, err := operationContext(op)
	if err != nil {
		return cResult("", err, errorOut)
	}
	result, err := mobile.HubInspect(ctx, C.GoString(reference), C.GoString(optionsJSON))
	return cResult(result, err, errorOut)
}

//export gopherllm_hub_download
func gopherllm_hub_download(reference, filesJSON, optionsJSON *C.char, op uintptr, onProgress C.gopherllm_hub_progress_cb, userdata unsafe.Pointer, errorOut **C.char) *C.char {
	ctx, err := operationContext(op)
	if err != nil {
		return cResult("", err, errorOut)
	}
	var progress huggingface.ProgressFunc
	if onProgress != nil {
		progress = func(event huggingface.Progress) { hubProgress(onProgress, userdata, event) }
	}
	result, err := mobile.HubDownload(ctx, C.GoString(reference), C.GoString(filesJSON), C.GoString(optionsJSON), progress)
	return cResult(result, err, errorOut)
}
