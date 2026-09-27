# GopherLLM for Swift

A Swift package for running GGUF models on the device in iOS and macOS apps,
with an async/await API over GopherLLM's C ABI.

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

| Source | Contents |
|---|---|
| `Sources/GopherLLM/GopherLLMEngine.swift` | The engine: load, generate, chat, stream, cancel, inspect, count tokens |
| `Sources/GopherLLM/Types.swift` | Options, messages, results and model info |
| `Tests/GopherLLMTests/` | End-to-end tests through the XCFramework |
