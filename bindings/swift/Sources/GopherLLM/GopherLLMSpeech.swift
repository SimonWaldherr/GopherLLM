import Foundation
import GopherLLMCore

public enum SpeechBackend: String, Codable, Sendable, CaseIterable {
    case voxtral, phonon, whisper

    /// Availability depends on the XCFramework build and current platform.
    public var isAvailable: Bool { gopherllm_speech_available(rawValue) != 0 }
}

public struct SpeechOptions: Encodable, Sendable {
    public var backend: SpeechBackend
    /// Reserved for future backends. Voxtral and Phonon detect the language.
    public var language: String?
    /// Run Voxtral without Metal.
    public var cpuOnly: Bool?

    public init(backend: SpeechBackend, language: String? = nil, cpuOnly: Bool? = nil) {
        self.backend = backend
        self.language = language
        self.cpuOnly = cpuOnly
    }
}

public struct SpeechTranscript: Decodable, Sendable, Equatable {
    /// The transcript, or only the command when an activation phrase is used.
    public let text: String
    /// Nil for push-to-talk. An unmatched clip has empty text and false here.
    public let matched: Bool?
}

/// A resident local speech model. Decodes audio in process on a background
/// queue and never accesses a microphone or the network.
public final class GopherLLMSpeech: Sendable {
    public static let sampleRate = 16_000
    public static let maxSamples = 30 * sampleRate

    private let handle: gopherllm_speech

    private init(handle: gopherllm_speech) { self.handle = handle }
    deinit { gopherllm_speech_free(handle) }

    /// Loads a local model once. Keep the recognizer alive across recordings.
    public static func open(modelAt url: URL, options: SpeechOptions) async throws -> GopherLLMSpeech {
        guard url.isFileURL else { throw GopherLLMError("Speech requires a local model file") }
        let optionsJSON = try JSON.encode(options)
        return try await NativeOperation.run { operation in
            var error: UnsafeMutablePointer<CChar>?
            let handle = gopherllm_speech_open(url.path, optionsJSON, operation, &error)
            if let message = takeString(error) { throw GopherLLMError(message) }
            guard handle != 0 else { throw GopherLLMError("Opening speech recognizer failed") }
            return GopherLLMSpeech(handle: handle)
        }
    }

    /// Borrows normalized mono float32 PCM at 16 kHz without converting it.
    /// Task cancellation applies only to this queued or running transcription.
    public func transcribe(_ samples: [Float], activationPhrase: String? = nil) async throws -> SpeechTranscript {
        guard !samples.isEmpty, samples.count <= Self.maxSamples else {
            throw GopherLLMError("Speech requires 1 to \(Self.maxSamples) samples at \(Self.sampleRate) Hz")
        }
        guard activationPhrase?.contains("\0") != true else { throw GopherLLMError("Activation phrase cannot contain NUL") }
        return try await NativeOperation.run { [self] operation in
            let json = try samples.withUnsafeBufferPointer { buffer in
                try withOptionalCString(activationPhrase) { phrase in
                    try check { gopherllm_speech_transcribe(handle, buffer.baseAddress!, UInt32(buffer.count), phrase, operation, $0) }
                }
            }
            return try JSON.decode(SpeechTranscript.self, from: json)
        }
    }
}
