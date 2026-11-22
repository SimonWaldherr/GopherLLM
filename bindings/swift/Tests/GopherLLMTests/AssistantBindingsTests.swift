import Foundation
import XCTest
@testable import GopherLLM

final class AssistantBindingsTests: XCTestCase {
    func testHubUsesItsExplicitOfflineCache() async throws {
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: cache) }
        let model = cache.appendingPathComponent("models--test--voice", isDirectory: true)
        let refs = model.appendingPathComponent("refs", isDirectory: true)
        let snapshot = model.appendingPathComponent("snapshots/fixture", isDirectory: true)
        try FileManager.default.createDirectory(at: refs, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try Data("fixture".utf8).write(to: refs.appendingPathComponent("main"))
        try Data("model".utf8).write(to: snapshot.appendingPathComponent("model.fermion"))
        try Data("{}".utf8).write(to: snapshot.appendingPathComponent("config.json"))

        let hub = GopherLLMHub(cacheDirectory: cache, offline: true)
        let manifest = try await hub.inspect("test/voice")
        XCTAssertEqual(manifest.repository, "test/voice")
        XCTAssertEqual(manifest.commit, "fixture")
        XCTAssertEqual(manifest.files.map(\.path), ["config.json", "model.fermion"])
        let files = try await hub.download("test/voice", files: ["model.fermion", "config.json"])
        XCTAssertEqual(files.map(\.lastPathComponent), ["model.fermion", "config.json"])
        XCTAssertEqual(try Data(contentsOf: files[0]), Data("model".utf8))

        do {
            _ = try await hub.download("test/voice", files: ["model.py"])
            XCTFail("An executable artifact was accepted")
        } catch let error as GopherLLMError {
            XCTAssertTrue(error.message.contains("unsupported model artifact"))
        }
    }

    func testSpeechAvailabilityAndFailure() async throws {
        XCTAssertTrue(SpeechBackend.voxtral.isAvailable)
        XCTAssertTrue(SpeechBackend.phonon.isAvailable)
        XCTAssertTrue(SpeechBackend.whisper.isAvailable)
        do {
            let path = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).gguf")
            _ = try await GopherLLMSpeech.open(modelAt: path, options: .init(backend: .voxtral))
            XCTFail("Opening a missing model unexpectedly succeeded")
        } catch let error as GopherLLMError {
            XCTAssertTrue(error.message.contains("Voxtral"))
        }
    }

    func testRealSpeechModelsWhenConfigured() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let pcmPath = env["GOPHERLLM_SWIFT_SPEECH_PCM"] else {
            throw XCTSkip("Set GOPHERLLM_SWIFT_SPEECH_PCM to a 16 kHz mono PCM16 test recording")
        }
        let audio = try Data(contentsOf: URL(fileURLWithPath: pcmPath))
        let samples = audio.withUnsafeBytes { raw in
            (0..<(raw.count / 2)).map {
                Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 2, as: Int16.self))) / 32768
            }
        }
        let models = SpeechBackend.allCases.compactMap { backend -> (SpeechBackend, String)? in
            env["GOPHERLLM_SWIFT_\(backend.rawValue.uppercased())_MODEL"].map { (backend, $0) }
        }
        guard !models.isEmpty else { throw XCTSkip("Set at least one GOPHERLLM_SWIFT_<BACKEND>_MODEL") }

        for (backend, path) in models {
            XCTAssertTrue(backend.isAvailable, "\(backend) is not compiled into this XCFramework")
            let recognizer = try await GopherLLMSpeech.open(modelAt: URL(fileURLWithPath: path),
                                                         options: .init(backend: backend))
            let match = try await recognizer.transcribe(samples, activationPhrase: "Computer")
            XCTAssertEqual(match.matched, true)
            XCTAssertTrue(match.text.lowercased().contains("wohnzimmer"), "\(backend): \(match.text)")
            let start = ContinuousClock.now
            let warm = try await recognizer.transcribe(samples, activationPhrase: "Computer")
            let elapsed = ContinuousClock.now - start
            XCTAssertEqual(warm, match)
            print("\(backend.rawValue) warmed Swift transcription: \(elapsed) — \(warm.text)")
            let miss = try await recognizer.transcribe(samples, activationPhrase: "Offset")
            XCTAssertEqual(miss.matched, false)
            XCTAssertEqual(miss.text, "")
        }
    }
}
