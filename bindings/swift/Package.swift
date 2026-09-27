// swift-tools-version:5.9
import PackageDescription

// GopherLLMCore.xcframework is a build product, not checked in: create it
// with `make xcframework` (scripts/build-xcframework.sh) before building this
// package.
let package = Package(
    name: "GopherLLM",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "GopherLLM", targets: ["GopherLLM"]),
    ],
    targets: [
        .binaryTarget(name: "GopherLLMCore", path: "GopherLLMCore.xcframework"),
        .target(
            name: "GopherLLM",
            dependencies: ["GopherLLMCore"],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Foundation"),
                .linkedFramework("Metal"),
            ]
        ),
        .testTarget(name: "GopherLLMTests", dependencies: ["GopherLLM"]),
    ]
)
