# GopherLLM for Swift

A Swift package for local text generation and speech recognition in iOS and
macOS apps, with an async/await API over GopherLLM's C ABI.

```swift
import GopherLLM

let engine = GopherLLMEngine()
try await engine.load(modelAt: modelURL)
for try await event in engine.stream([.user("Hello!")]) {
    if case .delta(let text) = event { print(text, terminator: "") }
}
```

`GopherLLMCore.xcframework`, the package's binary part, is built from the Go
sources and not checked in. From the repository root:

```sh
make xcframework   # writes bindings/swift/GopherLLMCore.xcframework
make swift-test    # runs Tests/ against a synthetic model on macOS
```

Then add this directory to an Xcode project as a local package. The full
guide, including model delivery, memory planning and entitlements, is
[docs/ios.md](../../docs/ios.md).

## Speech and model downloads

Keep one `GopherLLMSpeech` instance per loaded model. Pass normalized mono
float32 samples at 16 kHz. The app owns microphone permission, recording and
resampling. The optional activation phrase returns only the recognized
command. A clip that does not match exposes empty text. Canceling a Swift
task stops its own operation, including a request waiting for the model.

```swift
let recognizer = try await GopherLLMSpeech.open(modelAt: speechModelURL,
                                               options: .init(backend: .phonon))
let transcript = try await recognizer.transcribe(samples, activationPhrase: "Computer")
if transcript.matched == true, !transcript.text.isEmpty {
    // Forward the recognized command to the assistant.
}
```

`GopherLLMHub` searches and inspects Hugging Face repositories and downloads
selected model files to the directory chosen by the app. A text model can
use a GGUF selector with automatic shard resolution. For Phonon, select the
extracted weight file together with its sibling `config.json`. The app can
pass a token from its Keychain. The native client sends it to the Hub origin
and keeps it out of redirected CDN requests. `offline: true` opens the existing
cache. Download callbacks report per-file progress and may run on worker threads.

Use `SpeechBackend.isAvailable` to show the supported backends. Voxtral uses
GGUF, and Phonon expects an extracted `model.fermion` beside `config.json`.
Whisper accepts a legacy GGML F16/F32 checkpoint such as `ggml-base.bin`.
Set `SpeechOptions.language` to a language code for Whisper or leave it
unset for automatic detection. `SpeechOptions.cpuOnly` selects the CPU for Voxtral. The
[Hestia macOS app](../../apps/HestiaMac/README.md) uses a macOS build of the
same framework that also embeds the assistant runtime.

| Source | Contents |
|---|---|
| `Sources/GopherLLM/GopherLLMEngine.swift` | The engine: load, generate, chat, stream, cancel, inspect, count tokens |
| `Sources/GopherLLM/GopherLLMSpeech.swift` | Resident speech recognition and activation phrase matching |
| `Sources/GopherLLM/GopherLLMHub.swift` | Hugging Face model search, manifests and downloads |
| `Sources/GopherLLM/Types.swift` | Options, messages, results and model info |
| `Tests/GopherLLMTests/` | End-to-end tests through the XCFramework |
