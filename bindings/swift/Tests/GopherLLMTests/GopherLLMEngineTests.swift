import XCTest
import GopherLLM

/// End-to-end tests through the XCFramework against internal/testmodel's
/// synthetic GGUF. `make swift-test` writes it and sets GOPHERLLM_TEST_MODEL;
/// without that variable the model-backed tests are skipped.
final class GopherLLMEngineTests: XCTestCase {
    /// What the synthetic model generates for "Hallo" with `deterministic`;
    /// the Go, C, Rust and Python tests assert the same string.
    static let expectedText = "$(F=bBz\".C\\[}3WeO`>i[\"cZ"
    static let deterministic = GenerationOptions(maxTokens: 24, seed: 1)

    private func modelURL() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["GOPHERLLM_TEST_MODEL"], !path.isEmpty else {
            throw XCTSkip("GOPHERLLM_TEST_MODEL is not set; run `make swift-test`")
        }
        return URL(fileURLWithPath: path)
    }

    private func loadedEngine() async throws -> GopherLLMEngine {
        let engine = GopherLLMEngine()
        try await engine.load(modelAt: try modelURL(), options: LoadOptions(threads: 1))
        return engine
    }

    func testLibraryInfo() throws {
        XCTAssertFalse(GopherLLMEngine.version.isEmpty)
        let runtime = try GopherLLMEngine.runtimeInfo()
        XCTAssertEqual(runtime.version, GopherLLMEngine.version)
        XCTAssertGreaterThan(runtime.cpus, 0)
    }

    func testInspectModel() async throws {
        let info = try await GopherLLMEngine.inspectModel(at: try modelURL())
        XCTAssertEqual(info.name, "gopherllm-synth-testmodel")
        XCTAssertEqual(info.architecture, "llama")
        XCTAssertTrue(info.supported)
        XCTAssertEqual(info.quantization, "F32")
        XCTAssertGreaterThan(info.kvCacheBytesPerToken, 0)
        XCTAssertNil(info.loadTimeMs)
        do {
            _ = try await GopherLLMEngine.inspectModel(at: URL(fileURLWithPath: "/nonexistent/model.gguf"))
            XCTFail("inspecting a missing file succeeded")
        } catch is GopherLLMError {}
    }

    func testGenerateChatAndStream() async throws {
        let engine = try await loadedEngine()
        XCTAssertTrue(engine.isLoaded)
        XCTAssertEqual(engine.modelName, "gopherllm-synth-testmodel")
        let info = try XCTUnwrap(try engine.info())
        XCTAssertEqual(info.vocabSize, 98)
        XCTAssertNotNil(info.loadTimeMs)

        let text = try await engine.generate("Hallo", options: Self.deterministic)
        XCTAssertEqual(text, Self.expectedText)

        let result = try await engine.chat([.system("Be brief."), .user("Hallo")], options: Self.deterministic)
        XCTAssertEqual(result.finishReason, "length")
        XCTAssertEqual(result.generatedTokens, 24)
        XCTAssertGreaterThan(result.promptTokens, 0)

        var streamed = ""
        var completed: GenerationResult?
        for try await event in engine.stream([.user("Hallo")], options: Self.deterministic) {
            switch event {
            case .delta(let piece): streamed += piece
            case .completed(let result): completed = result
            }
        }
        XCTAssertEqual(streamed, Self.expectedText)
        XCTAssertEqual(completed?.text, Self.expectedText)

        let tokens = try await engine.countTokens("Hallo")
        XCTAssertEqual(tokens, 6)
    }

    func testEndingAStreamEarlyLeavesTheEngineUsable() async throws {
        let engine = try await loadedEngine()
        var deltas = 0
        for try await event in engine.stream([.user("Hallo")], options: GenerationOptions(maxTokens: 200)) {
            if case .delta = event { deltas += 1 }
            if deltas == 3 { break }
        }
        XCTAssertEqual(deltas, 3)

        let cancelled = Task { try await engine.chat([.user("Hallo")], options: GenerationOptions(maxTokens: 200)) }
        cancelled.cancel()
        _ = await cancelled.result // finished before the cancel landed, or failed as canceled

        let text = try await engine.generate("Hallo", options: Self.deterministic)
        XCTAssertEqual(text, Self.expectedText)
    }

    func testErrors() async throws {
        let engine = GopherLLMEngine()
        XCTAssertFalse(engine.isLoaded)
        XCTAssertNil(try engine.info())
        do {
            _ = try await engine.generate("Hallo")
            XCTFail("generated without a model")
        } catch let error as GopherLLMError {
            XCTAssertTrue(error.message.contains("not loaded"), error.message)
        }
        do {
            try await engine.load(modelAt: URL(fileURLWithPath: "/nonexistent/model.gguf"))
            XCTFail("loaded a missing file")
        } catch is GopherLLMError {}

        let loaded = try await loadedEngine()
        do {
            try await loaded.load(modelAt: try modelURL())
            XCTFail("loaded a second model into one engine")
        } catch is GopherLLMError {}
        try await loaded.unload()
        XCTAssertFalse(loaded.isLoaded)
    }
}
