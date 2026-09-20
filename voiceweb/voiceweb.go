// Package voiceweb holds the one browser-side audio-capture building block
// that is genuinely generic enough to share between GopherLLM's own chat
// web UI (server/web_ui) and other GopherLLM-based applications, such as
// cmd/hestia: an AudioWorkletProcessor that resamples the microphone to
// 16kHz mono PCM16 and posts it back in ~200ms blocks. It has no
// dependency on any particular page's DOM, styling, or HTTP contract, so
// it can be embedded byte-for-byte by any caller.
//
// The higher-level orchestration logic each UI builds on top of this
// worklet (server/web_ui/audio.js) is presentation- and endpoint-specific
// (its own DOM element IDs, its own REST paths under /v1/audio/...) and is
// deliberately NOT part of this package -- see CONCEPT.md in
// cmd/hestia: "Wiederverwendbare Web-Audiologik wird beim Implementieren
// aus der bisherigen UI herausgelöst; beide UIs behalten ihre eigenen
// Bedienelemente und Tests." Each caller wires the worklet's postMessage
// events into its own UI and endpoints; only the capture/resample logic
// is shared.
package voiceweb

import _ "embed"

// AudioWorkletJS is the source of the "voxtral-pcm" AudioWorkletProcessor.
// A caller registers it via AudioContext.audioWorklet.addModule and talks
// to it over MessagePort: send "stop" to flush and stop; it posts
// ArrayBuffer PCM16 chunks, then the string "stopped" once the final
// partial block (if any) has been posted.
//
//go:embed audio-worklet.js
var AudioWorkletJS string
