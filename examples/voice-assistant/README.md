# Gopher Voice

Gopher Voice is a local browser demo for spoken conversations. It accepts
microphone recordings, audio files and typed messages. A speech model
transcribes audio, Ministral answers, and the browser plays the reply.
Recent turns stay in the tab so you can ask follow-up questions.

## Run with Phonon and Ministral

Use a local Ministral Instruct GGUF and the extracted
[Phonon 2.5 checkpoint](../../docs/phonon.md).

```sh
go run -tags metal ./examples/voice-assistant \
  -metal \
  -model /path/to/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
  -phonon /path/to/model.fermion
```

Open [Gopher Voice](http://127.0.0.1:8092). Click **Start microphone** and
grant browser permission. A pause ends the turn automatically, or you
can click **Stop & send**. Check **Hands-free follow-ups** before a
microphone turn to reopen the mic after playback. **New conversation**
stops the current turn and clears the conversation. Recording starts from
the button. The mic is closed during transcription, generation and
playback.

On CPU, omit `-tags metal` and `-metal`. The browser uses a local system
voice when one is available. For a German WAV reply synthesized by
GopherLLM, add `-tts /path/to/mms-tts-deu` and use the
[native MMS model setup](../../docs/generation.md#german-speech).
The loopback demo accepts clips up to 30 seconds and keeps the latest four
complete turns as model context.

## Run with Voxtral

Replace `-phonon /path/to/model.fermion` with
`-voxtral /path/to/voxtral-realtime.gguf` to use GopherLLM's Voxtral backend.
Use `-whisper /path/to/ggml-base.bin` for native Whisper with automatic
language detection.

## Local measurements

For Ministral, `TestMetalFixedDecodeBenchmark` uses a fixed 160-token
prefix and then runs 64 steps with full logits. Each figure is the
median of three warm runs on the same M2 Max. The Metal matrix kernel
loads contiguous activation vectors and lets the SIMD-group load
transpose the shared tile.

| Ministral Q4_K_M | Prefill before → after | Decode before → after |
| --- | ---: | ---: |
| 3B Instruct | 454 → 593 tok/s | 89.3 → 88.9 tok/s |
| 14B Reasoning | 121 → 158 tok/s | 26.9 → 26.8 tok/s |

Each model had bit-identical final logits before and after the kernel
change.

```sh
go test -tags metal -c -o /tmp/engine-metal.test ./internal/engine
GOPHERLLM_FIXED_MODEL=/path/to/ministral.gguf \
  /tmp/engine-metal.test -test.run '^TestMetalFixedDecodeBenchmark$' -test.v
```
