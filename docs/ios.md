# iOS integration

GopherLLM runs GGUF models on the device, inside the app process: no server,
no network, no runtime besides the app itself. An iOS app uses it through a
Swift package whose binary part is an XCFramework built from GopherLLM's C ABI:

```
your app ─► GopherLLM (Swift package, bindings/swift)
              └─► GopherLLMCore.xcframework ─► C ABI (gopherllm.h) ─► mobile.Engine ─► inference engine (Go, NEON, optional Metal)
```

The same package also works in macOS apps.

## Requirements

- macOS with Xcode 15 or newer, and Go 1.25 or newer, to build the XCFramework.
- Deployment targets iOS 17 and macOS 14 (change them with
  `IOS_DEPLOYMENT_TARGET` / `MACOS_DEPLOYMENT_TARGET` when building the
  XCFramework and in `bindings/swift/Package.swift`).

## 1. Build the XCFramework

```sh
make xcframework
```

This runs `scripts/build-xcframework.sh`, which cross-compiles the C ABI as a
static library for iOS devices (arm64), the simulator (arm64 and x86_64) and
macOS (arm64 and x86_64), and writes
`bindings/swift/GopherLLMCore.xcframework`. Only Go and Xcode are involved; no
gomobile or other tool is downloaded. Useful variables:

| Variable | Effect |
|---|---|
| `PLATFORMS="ios ios-simulator"` | Skip slices you do not need (faster local builds) |
| `METAL=0` | Leave the Metal GPU kernels out |
| `XCFRAMEWORK_OUT=/path/GopherLLMCore.xcframework` | Write it elsewhere |

The XCFramework is a build product and is not checked in. Rebuild it after
changing Go code; the Swift package picks it up on the next build.

## 2. Add the Swift package to your app

In Xcode: **File ▸ Add Package Dependencies… ▸ Add Local…**, choose
`bindings/swift` in your GopherLLM checkout, and add the **GopherLLM** library
to your app target. From another package:

```swift
.package(path: "../GopherLLM/bindings/swift")
```

The module map links Accelerate, Foundation and Metal automatically.

To ship the package without a GopherLLM checkout (for example from your own
repository), zip the XCFramework, compute its checksum with
`swift package compute-checksum GopherLLMCore.xcframework.zip`, and replace the
`binaryTarget(path:)` in `Package.swift` with `binaryTarget(url:checksum:)`
pointing at the uploaded zip.

## 3. Use it

```swift
import GopherLLM

let engine = GopherLLMEngine()

// Check a file before committing memory to it: reads only the GGUF header.
let info = try await GopherLLMEngine.inspectModel(at: modelURL)
guard info.supported else { throw MyError.unsupported(info.architecture) }

try await engine.load(modelAt: modelURL, options: LoadOptions(metal: true))

// Stream a reply to a conversation.
var history: [ChatMessage] = [.system("You are concise."), .user("What is GGUF?")]
var reply = ""
for try await event in engine.stream(history, options: GenerationOptions(maxTokens: 256, contextWindowMode: .recent)) {
    switch event {
    case .delta(let text): reply += text             // update the UI
    case .completed(let result): print(result.tokensPerSecond)
    }
}
history.append(.assistant(reply))

// Or wait for the whole answer.
let result = try await engine.chat(history + [.user("Shorter, please.")])
```

- **Conversations.** Pass the whole history each time. With
  `contextWindowMode: .recent` the oldest complete turns are dropped once the
  conversation outgrows the model's context (`.autoCompress` condenses them
  first); `result.contextWindow` reports what was dropped. The default,
  `.full`, fails instead.
- **Cancellation.** Cancelling the Swift task, or breaking out of the stream
  loop, stops generation. `engine.cancel()` stops the running request
  directly.
- **Threading.** All methods may be called from any actor. Model work runs on
  a background queue, one request at a time per engine, in order.
  `isLoaded`, `modelName` and `info()` return immediately even during a
  generation; `countTokens` runs alongside it.
- **Options.** Every option left `nil` keeps GopherLLM's default. A fixed
  `seed` makes sampling reproducible; `temperature: 0` is greedy decoding;
  `jsonObject: true` constrains the reply to one JSON object.

## Getting models onto the device

GGUF files are usually hundreds of megabytes to several gigabytes, too large
to bundle with the app. Common approaches:

- Let the user import a file (`.fileImporter`) and copy it into Application
  Support, as the demo app does. Access to the picked file is
  security-scoped: call `startAccessingSecurityScopedResource()` around the
  copy.
- Download it with a background `URLSession` download task (for example from
  Hugging Face) into Application Support. The mobile library itself contains
  no network code.
- Exclude model files from backups:
  `var values = URLResourceValues(); values.isExcludedFromBackup = true; try url.setResourceValues(values)`.

Small quantizations (Q4_K_M and similar) of small models are the practical
choice on phones.

## Memory

iOS terminates apps that exceed their memory limit, so plan the budget before
loading:

- **Weights.** Quantized tensors are memory-mapped and used in place, so they
  are file-backed pages the system can page out and back in. F16/F32/BF16
  tensors are expanded into owned memory unless the model is loaded with
  `LoadOptions(outOfCore: true)`. `ModelInfo.tensorBytes` is the weights' size
  and `ModelInfo.quantization` their dominant type.
- **Attention (KV) cache.** It grows with the conversation:
  `ModelInfo.kvCacheBytesPerToken` × (prompt + reply tokens). Limit
  `maxTokens` and use `contextWindowMode: .recent` to bound it.
- **Entitlements.** Mapping a multi-gigabyte file needs address space beyond
  an app's default. Add `com.apple.developer.kernel.extended-virtual-addressing`
  and, to raise the memory limit on devices that allow it,
  `com.apple.developer.kernel.increased-memory-limit` (enable both
  capabilities for the App ID). `os_proc_available_memory()` reports what the
  app may still allocate.
- **Pressure.** Call `engine.unload()` on
  `UIApplication.didReceiveMemoryWarningNotification`, and consider unloading
  when the app moves to the background. The engine stays usable; load again
  when needed.

## Performance

- **Metal.** `LoadOptions(metal: true)` runs eligible layers of supported
  layouts (Qwen, Ministral and Gemma families; see the README's Performance
  Notes) on the GPU and falls back to the CPU elsewhere.
  `GopherLLMEngine.runtimeInfo().metalAvailable` says whether the build has it.
- **Threads.** The default picks a worker count automatically. `threads` is
  process-wide; measure `GenerationResult.tokensPerSecond` on real devices
  before changing it.
- **Build configuration.** The Go code in the XCFramework is always compiled
  with optimizations, so Debug builds of the app infer at full speed.
- **Heat.** Sustained generation makes phones throttle; expect lower tokens/s
  in long sessions than in short benchmarks.

## Shipping

- GopherLLM is ordinary native code: no JIT, no downloaded code. Metal kernels
  are compiled at run time from source embedded in the library, which iOS
  permits.
- Everything runs on the device and the library opens no network
  connections, which keeps privacy declarations simple.
- The Go runtime and inference engine add several megabytes to the app
  binary.

## The demo app

`examples/ios/GopherLLMDemo` is a SwiftUI chat app on the Swift package:
import a GGUF, see its details before loading, chat with streaming and a
running history, stop generation, and tune sampling. Build the XCFramework,
then open `GopherLLMDemo.xcodeproj` and run the **GopherLLMDemo** scheme.

## Validation

| Command | Checks |
|---|---|
| `make capi-test` | The C ABI end to end on a synthetic model (Linux or macOS) |
| `make swift-test` | The Swift package's tests against the same model (macOS) |
| `make ios-demo-build` | The demo app builds for the simulator and for devices |
| `make ios-check` | All of the above plus `go test` and `go vet` |

CI runs these on every push.

## Current limits

- The mobile API covers text generation and chat. Vision (Pixtral), tool
  calling, embeddings, speech (Voxtral, Parakeet) and Laya classification are
  available to Go programs but not yet exposed through the C ABI and Swift.
  Adding one follows [the embedding layers](PROJECT_STRUCTURE.md#embedding-layers).
- The KV cache uses f32 on ARM64 by default. The half-size f16 cache is only
  switchable with the `GOPHERLLM_KV_F16=1` environment variable, which must be
  set before the app launches (for example in the Xcode scheme).
- The Apple Neural Engine is not used.
