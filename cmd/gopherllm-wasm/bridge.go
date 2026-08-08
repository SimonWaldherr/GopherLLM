//go:build js

package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"syscall/js"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

var (
	runnerMu sync.Mutex
	runner   *gopherllm.Runner

	// generating guards against two overlapping gopherllm_generate calls.
	// Runner.GenerateChatStreamUntil already serializes on the Runner's own
	// genLock, so a second concurrent call would not race -- it would just
	// block silently until the first one finishes, with no signal to the
	// caller about why its promise appears to hang. Rejecting immediately
	// instead makes that state visible.
	generating atomic.Bool

	// currentStop is the stop flag for whichever generation is presently
	// running (nil when none is). Each gopherllm_generate call creates its
	// own *atomic.Bool and only that call's onToken closure ever reads it --
	// a single shared flag would let a second generate() call's startup
	// (which needs to arm a fresh, unset flag) silently clear a stop signal
	// that gopherllm_stopGeneration had just sent for a still-draining
	// earlier call on the same runner.
	currentStop atomic.Pointer[atomic.Bool]
)

func registerCallbacks() {
	js.Global().Set("gopherllm_loadModel", js.FuncOf(jsLoadModel))
	js.Global().Set("gopherllm_loadModelWithVision", js.FuncOf(jsLoadModelWithVision))
	js.Global().Set("gopherllm_hasVision", js.FuncOf(jsHasVision))
	js.Global().Set("gopherllm_generate", js.FuncOf(jsGenerate))
	js.Global().Set("gopherllm_stopGeneration", js.FuncOf(jsStopGeneration))
	js.Global().Set("gopherllm_isGenerating", js.FuncOf(jsIsGenerating))
	js.Global().Set("gopherllm_isModelLoaded", js.FuncOf(jsIsModelLoaded))
	js.Global().Set("gopherllm_webgpuSmokeTest", js.FuncOf(jsWebGPUSmokeTest))
	js.Global().Set("gopherllm_webgpuKernelTest", js.FuncOf(jsWebGPUKernelTest))
	js.Global().Set("gopherllm_webgpuStatus", js.FuncOf(jsWebGPUStatus))
	js.Global().Set("gopherllm_setWebGPUForceDisabled", js.FuncOf(jsSetWebGPUForceDisabled))
}

// jsSetWebGPUForceDisabled(disabled: boolean) => undefined
//
// Lets the demo/benchmark harness opt back out of GPU acceleration before
// the next gopherllm_loadModel call, for an A/B throughput comparison
// against the CPU path on the exact same bytes without a second wasm
// build. No effect on a Runner already loaded.
func jsSetWebGPUForceDisabled(this js.Value, args []js.Value) any {
	disabled := len(args) > 0 && args[0].Truthy()
	gopherllm.SetWebGPUForceDisabled(disabled)
	return nil
}

// jsIsGenerating() => boolean
func jsIsGenerating(this js.Value, args []js.Value) any {
	return generating.Load()
}

// jsIsModelLoaded() => boolean
func jsIsModelLoaded(this js.Value, args []js.Value) any {
	runnerMu.Lock()
	defer runnerMu.Unlock()
	return runner != nil
}

// requestStopCurrent signals whatever generation is presently running, if
// any. Safe to call when none is: currentStop is nil until the first
// gopherllm_generate call, and a stale (already-finished) flag is a
// harmless no-op to set.
func requestStopCurrent() {
	if s := currentStop.Load(); s != nil {
		s.Store(true)
	}
}

// jsWebGPUStatus() => Promise<string>
//
// Reports whether gopherllm.WebGPUAvailable() sees a usable device -- a
// diagnostic for confirming a generation call actually took the GPU-backed
// Weight.MatvecInto path rather than silently falling back to CPU.
//
// Must be a Promise, not a direct return: WebGPUAvailable's first-ever call
// blocks synchronously on navigator.gpu.requestAdapter()/requestDevice(),
// which only resolve via a JS microtask. A js.FuncOf callback invoked
// directly (not wrapped in newPromise, i.e. not run on its own goroutine)
// runs on the same goroutine JS used to call it; blocking that goroutine
// prevents the JS call from ever returning, so the Promise's microtask can
// never run either -- Go's wasm scheduler detects every goroutine asleep
// with nothing left runnable and fatally exits the whole wasm instance
// (confirmed by an earlier version of this function doing exactly that).
func jsWebGPUStatus(this js.Value, args []js.Value) any {
	return newPromise(func(resolve, reject js.Value) {
		if gopherllm.WebGPUAvailable() {
			resolve.Invoke("available")
			return
		}
		resolve.Invoke("unavailable")
	})
}

// jsWebGPUKernelTest() => Promise<string>
//
// Compares the hand-written Q4_K/Q6_K WGSL matvec kernels against this
// project's existing portable Go reference implementation -- see
// webgpu_kernel_test.go.
func jsWebGPUKernelTest(this js.Value, args []js.Value) any {
	return newPromise(func(resolve, reject js.Value) {
		msg, err := runWebGPUKernelTest(context.Background())
		if err != nil {
			reject.Invoke(err.Error())
			return
		}
		resolve.Invoke(msg)
	})
}

// jsWebGPUSmokeTest() => Promise<string>
//
// Validates the internal/webgpu plumbing (device acquisition, buffer
// upload, shader compilation, dispatch, readback) independent of any model
// loading -- see webgpu_smoke.go.
func jsWebGPUSmokeTest(this js.Value, args []js.Value) any {
	return newPromise(func(resolve, reject js.Value) {
		msg, err := runWebGPUSmokeTest(context.Background())
		if err != nil {
			reject.Invoke(err.Error())
			return
		}
		resolve.Invoke(msg)
	})
}

// jsLoadModel(bytes: Uint8Array) => Promise<boolean>
//
// bytes is copied into Go-owned memory immediately (js.CopyBytesToGo), so
// the caller is free to drop its own reference to the source ArrayBuffer as
// soon as this call returns — important for staying under wasm's 32-bit
// address space with a multi-hundred-MB model.
func jsLoadModel(this js.Value, args []js.Value) any {
	if len(args) < 1 || args[0].IsUndefined() || args[0].IsNull() {
		return rejectPromise(fmt.Errorf("gopherllm_loadModel: expected a Uint8Array argument"))
	}
	src := args[0]
	data := make([]byte, src.Get("length").Int())
	js.CopyBytesToGo(data, src)

	return newPromise(func(resolve, reject js.Value) {
		r, err := gopherllm.RunnerFromGGUFBytesWithOptions(data, gopherllm.LoadOptions{})
		if err != nil {
			reject.Invoke(err.Error())
			return
		}

		runnerMu.Lock()
		prev := runner
		runner = r
		runnerMu.Unlock()
		if prev != nil {
			// prev.Close() blocks on prev's own genLock, so it wouldn't
			// actually race with a generation still in flight against prev
			// -- but it would sit there until that generation runs to
			// completion on its own, which could be many seconds, with
			// nothing telling the caller their loadModel promise is
			// pending for that reason rather than stuck. Ask it to wind
			// down first so switching models stays responsive.
			requestStopCurrent()
			prev.Close()
		}

		resolve.Invoke(true)
	})
}

// jsLoadModelWithVision(textBytes: Uint8Array, visionBytes?: Uint8Array) => Promise<boolean>
//
// Like jsLoadModel, but optionally also loads a paired Pixtral-style vision
// projector GGUF (gopherllm.RunnerFromGGUFBytesWithVision), enabling
// ChatMessage.Images on subsequent gopherllm_generate calls. Passing
// undefined/null for visionBytes loads text-only, same as jsLoadModel.
func jsLoadModelWithVision(this js.Value, args []js.Value) any {
	if len(args) < 1 || args[0].IsUndefined() || args[0].IsNull() {
		return rejectPromise(fmt.Errorf("gopherllm_loadModelWithVision: expected a Uint8Array text-model argument"))
	}
	textSrc := args[0]
	textData := make([]byte, textSrc.Get("length").Int())
	js.CopyBytesToGo(textData, textSrc)

	var visionData []byte
	if len(args) > 1 && !args[1].IsUndefined() && !args[1].IsNull() {
		visionSrc := args[1]
		visionData = make([]byte, visionSrc.Get("length").Int())
		js.CopyBytesToGo(visionData, visionSrc)
	}

	return newPromise(func(resolve, reject js.Value) {
		var r *gopherllm.Runner
		var err error
		if visionData != nil {
			r, err = gopherllm.RunnerFromGGUFBytesWithVision(textData, visionData, gopherllm.LoadOptions{})
		} else {
			r, err = gopherllm.RunnerFromGGUFBytesWithOptions(textData, gopherllm.LoadOptions{})
		}
		if err != nil {
			reject.Invoke(err.Error())
			return
		}

		runnerMu.Lock()
		prev := runner
		runner = r
		runnerMu.Unlock()
		if prev != nil {
			requestStopCurrent()
			prev.Close()
		}

		resolve.Invoke(true)
	})
}

// jsHasVision() => boolean
//
// Synchronous (no WebGPU/device access involved, just a field check on the
// loaded Runner) -- safe to call directly without the newPromise wrapping
// jsWebGPUStatus needs.
func jsHasVision(this js.Value, args []js.Value) any {
	runnerMu.Lock()
	r := runner
	runnerMu.Unlock()
	return r != nil && r.HasVision()
}

// wireMessage/wireOptions are the JSON shapes gopherllm_generate accepts
// from JS — plain strings/numbers only, translated here into the Go API's
// real types (gopherllm.ChatRole is an int enum, not JSON-friendly as-is).
type wireMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	// Images is base64-encoded raw image bytes (PNG/JPEG), no "data:" prefix
	// -- capped at one per message by the underlying renderer (see
	// runtime.go's renderMistralInstMessages).
	Images []string `json:"images,omitempty"`
}

type wireOptions struct {
	MaxTokens     int     `json:"maxTokens"`
	Temperature   float32 `json:"temperature"`
	TopP          float32 `json:"topP"`
	TopK          int     `json:"topK"`
	MinP          float32 `json:"minP"`
	RepeatPenalty float32 `json:"repeatPenalty"`
	SystemPrompt  string  `json:"systemPrompt"`
}

func chatRoleFromWire(role string) (gopherllm.ChatRole, error) {
	switch role {
	case "system":
		return gopherllm.ChatRoleSystem, nil
	case "user":
		return gopherllm.ChatRoleUser, nil
	case "assistant":
		return gopherllm.ChatRoleAssistant, nil
	case "tool":
		return gopherllm.ChatRoleTool, nil
	default:
		return 0, fmt.Errorf("unknown chat role %q", role)
	}
}

// jsGenerate(messagesJSON: string, optionsJSON: string, onToken: (text: string) => boolean) => Promise<string>
//
// onToken is invoked once per generated token fragment; returning false from
// it (or a prior gopherllm_stopGeneration call) stops generation early, the
// same early-stop contract Runner.GenerateChatStreamUntil already exposes.
// The resolved value is the full generated text.
func jsGenerate(this js.Value, args []js.Value) any {
	if len(args) < 3 {
		return rejectPromise(fmt.Errorf("gopherllm_generate: expected (messagesJSON, optionsJSON, onToken)"))
	}
	messagesJSON := args[0].String()
	optionsJSON := args[1].String()
	onToken := args[2]

	var wireMessages []wireMessage
	if err := json.Unmarshal([]byte(messagesJSON), &wireMessages); err != nil {
		return rejectPromise(fmt.Errorf("gopherllm_generate: parsing messages: %w", err))
	}
	messages := make([]gopherllm.ChatMessage, len(wireMessages))
	for i, m := range wireMessages {
		role, err := chatRoleFromWire(m.Role)
		if err != nil {
			return rejectPromise(fmt.Errorf("gopherllm_generate: message %d: %w", i, err))
		}
		cm := gopherllm.ChatMessage{Role: role, Content: m.Content}
		for _, b64 := range m.Images {
			raw, err := base64.StdEncoding.DecodeString(b64)
			if err != nil {
				return rejectPromise(fmt.Errorf("gopherllm_generate: message %d: decoding image: %w", i, err))
			}
			cm.Images = append(cm.Images, gopherllm.ImageContent{Bytes: raw})
		}
		messages[i] = cm
	}

	opts := wireOptions{MaxTokens: 512, Temperature: 0.7, TopP: 0.9, TopK: 40, RepeatPenalty: 1.1}
	if optionsJSON != "" {
		if err := json.Unmarshal([]byte(optionsJSON), &opts); err != nil {
			return rejectPromise(fmt.Errorf("gopherllm_generate: parsing options: %w", err))
		}
	}
	genOptions := gopherllm.GenerationOptions{
		MaxTokens:    opts.MaxTokens,
		SystemPrompt: opts.SystemPrompt,
		Sampler: gopherllm.SamplerConfig{
			Temperature:   opts.Temperature,
			TopP:          opts.TopP,
			TopK:          opts.TopK,
			MinP:          opts.MinP,
			RepeatPenalty: opts.RepeatPenalty,
		},
	}

	runnerMu.Lock()
	r := runner
	runnerMu.Unlock()
	if r == nil {
		return rejectPromise(fmt.Errorf("gopherllm_generate: no model loaded; call gopherllm_loadModel first"))
	}

	// Runner.GenerateChatStreamUntil would serialize a second concurrent
	// call on its own genLock rather than race, but silently blocking with
	// no feedback reads as a hung promise from the JS side. Reject with a
	// clear reason instead; the caller can gopherllm_stopGeneration first
	// if that is what they actually wanted.
	if !generating.CompareAndSwap(false, true) {
		return rejectPromise(fmt.Errorf("gopherllm_generate: a generation is already in progress; call gopherllm_stopGeneration first"))
	}

	stop := new(atomic.Bool)
	currentStop.Store(stop)

	return newPromise(func(resolve, reject js.Value) {
		defer generating.Store(false)
		result, err := r.GenerateChatStreamUntil(messages, genOptions, func(token string) bool {
			if stop.Load() {
				return false
			}
			// onToken is a JS function; Invoke blocks the calling goroutine
			// until the JS call returns, which is fine here since it isn't
			// itself async (no Promise involved on this side).
			ret := onToken.Invoke(token)
			return ret.IsUndefined() || ret.Truthy()
		})
		if err != nil {
			reject.Invoke(err.Error())
			return
		}
		resolve.Invoke(result.Text)
	})
}

// jsStopGeneration() => undefined
//
// Signals whichever generation gopherllm_generate most recently started.
// A no-op if none is running.
func jsStopGeneration(this js.Value, args []js.Value) any {
	requestStopCurrent()
	return nil
}
