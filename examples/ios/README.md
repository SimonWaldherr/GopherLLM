# GopherLLM iOS demo

A SwiftUI chat app on the [GopherLLM Swift package](../../bindings/swift):
import a GGUF, check its details before loading, chat with streaming and a
running history, stop generation, and tune sampling. It unloads the model on
memory warnings.

```sh
make xcframework                 # from the repository root, on macOS
open examples/ios/GopherLLMDemo/GopherLLMDemo.xcodeproj
```

Run the **GopherLLMDemo** scheme on a simulator or device (set your team
under Signing & Capabilities for a device). Large models need the memory
entitlements described in [docs/ios.md](../../docs/ios.md#memory).
`make ios-demo-build` builds it headlessly for the simulator and for devices.
