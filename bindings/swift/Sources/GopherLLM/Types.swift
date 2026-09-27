import Foundation

/// An error reported by GopherLLM, with its message as the description.
public struct GopherLLMError: LocalizedError, Sendable, Equatable {
    public let message: String

    public init(_ message: String) { self.message = message }

    public var errorDescription: String? { message }
}

/// How a model is loaded. Every `nil` field keeps GopherLLM's default.
public struct LoadOptions: Encodable, Sendable, Equatable {
    /// How weights are read into memory before the first request.
    public enum Prefault: String, Encodable, Sendable {
        /// Page weights in on demand (the default): fastest load, least memory pressure.
        case none
        /// Read the always-used tensors up front.
        case core
        /// Read the whole file up front.
        case all
    }

    /// Compute threads; `nil` or 0 picks automatically. Process-wide.
    public var threads: Int?
    /// Run eligible layers on the GPU (see `GopherLLMEngine.runtimeInfo().metalAvailable`).
    public var metal: Bool?
    public var prefault: Prefault?
    public var prepareQuantized: Bool?
    public var outOfCore: Bool?

    public init(threads: Int? = nil, metal: Bool? = nil, prefault: Prefault? = nil,
                prepareQuantized: Bool? = nil, outOfCore: Bool? = nil) {
        self.threads = threads
        self.metal = metal
        self.prefault = prefault
        self.prepareQuantized = prepareQuantized
        self.outOfCore = outOfCore
    }
}

/// Sampling and length settings for one request. Every `nil` field keeps
/// GopherLLM's default (256 tokens, its default sampler and system prompt).
public struct GenerationOptions: Encodable, Sendable, Equatable {
    /// What happens when a conversation no longer fits the model's context.
    public enum ContextWindowMode: String, Encodable, Sendable {
        /// Fail the request (the default).
        case full
        /// Drop the oldest complete turns until it fits.
        case recent
        /// Condense older turns, then drop the oldest if still needed.
        case autoCompress
    }

    public var maxTokens: Int?
    /// 0 selects greedy decoding.
    public var temperature: Float?
    public var topP: Float?
    public var topK: Int?
    public var minP: Float?
    public var repeatPenalty: Float?
    /// A fixed non-zero seed makes sampling reproducible.
    public var seed: UInt64?
    /// Used when the conversation has no leading system message; "" for none.
    public var systemPrompt: String?
    public var stop: [String]?
    public var contextWindowMode: ContextWindowMode?
    /// Constrain the reply to a single JSON object.
    public var jsonObject: Bool?

    public init(maxTokens: Int? = nil, temperature: Float? = nil, topP: Float? = nil, topK: Int? = nil,
                minP: Float? = nil, repeatPenalty: Float? = nil, seed: UInt64? = nil,
                systemPrompt: String? = nil, stop: [String]? = nil,
                contextWindowMode: ContextWindowMode? = nil, jsonObject: Bool? = nil) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repeatPenalty = repeatPenalty
        self.seed = seed
        self.systemPrompt = systemPrompt
        self.stop = stop
        self.contextWindowMode = contextWindowMode
        self.jsonObject = jsonObject
    }
}

/// One turn of a conversation.
public struct ChatMessage: Codable, Sendable, Hashable {
    public enum Role: String, Codable, Sendable {
        case system, user, assistant
    }

    public var role: Role
    public var content: String

    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }

    public static func system(_ content: String) -> ChatMessage { ChatMessage(role: .system, content: content) }
    public static func user(_ content: String) -> ChatMessage { ChatMessage(role: .user, content: content) }
    public static func assistant(_ content: String) -> ChatMessage { ChatMessage(role: .assistant, content: content) }
}

/// A finished generation.
public struct GenerationResult: Decodable, Sendable, Equatable {
    public let text: String
    /// "stop" (natural end or stop sequence) or "length" (token or context limit).
    public let finishReason: String
    public let generatedTokens: Int
    /// Chain-of-thought the model emitted separately from `text`, if any.
    public let reasoningText: String?
    public let promptTokens: Int
    public let ttftMs: Double
    public let prefillMs: Double
    public let decodeMs: Double
    public let totalMs: Double
    public let tokensPerSecond: Double
    /// Set when `contextWindowMode` trimmed or condensed the conversation.
    public let contextWindow: ContextWindow?

    public struct ContextWindow: Decodable, Sendable, Equatable {
        public let mode: String
        public let contextLength: Int
        public let promptBudget: Int
        public let promptTokens: Int
        public let inputMessages: Int
        public let retainedMessages: Int
        public let droppedMessages: Int
        public let compressedMessages: Int
    }
}

/// One step of a streamed generation.
public enum StreamEvent: Sendable, Equatable {
    /// The next piece of generated text.
    case delta(String)
    /// The generation finished; always the last event.
    case completed(GenerationResult)
}

/// What a GGUF file says about its model. `GopherLLMEngine.inspectModel`
/// fills the header fields; `GopherLLMEngine.info` also the loaded ones.
public struct ModelInfo: Decodable, Sendable, Equatable {
    public let name: String
    public let architecture: String
    /// Whether this build can run the architecture.
    public let supported: Bool
    public let parameters: Int64
    /// Size of the weights: a lower bound for their memory once loaded.
    public let tensorBytes: Int64
    public let bitsPerWeight: Double
    /// The tensor type holding most of the weight bytes, e.g. "Q4_K".
    public let quantization: String
    public let contextLength: Int
    public let layers: Int
    public let vocabSize: Int
    public let chatTemplate: String
    /// Attention-cache growth per context token; multiply by prompt plus
    /// reply length to budget memory for a conversation.
    public let kvCacheBytesPerToken: Int64

    // Only set for a loaded model.
    public let fileSizeBytes: Int64?
    public let mapped: Bool?
    public let outOfCore: Bool?
    public let loadTimeMs: Int64?
    public let metalAvailable: Bool?
}

/// The library build and device, for diagnostics screens.
public struct RuntimeInfo: Decodable, Sendable, Equatable {
    public let version: String
    public let goos: String
    public let goarch: String
    public let cpus: Int
    public let metalAvailable: Bool
    /// "available", or why Metal is not.
    public let metalStatus: String
}

enum JSON {
    static func encode<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }

    static func decode<T: Decodable>(_ type: T.Type, from json: String) throws -> T {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(type, from: Data(json.utf8))
    }
}
