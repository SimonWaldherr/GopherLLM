import Foundation
import GopherLLMCore

/// A GopherLLM inference engine holding at most one loaded GGUF model.
///
/// All methods are safe to call from any thread or actor. Work that runs the
/// model happens on a background queue, never on the caller's thread;
/// requests on one engine run one at a time, in order. Cancelling the Swift
/// task awaiting a request (or ending iteration of a stream) stops it.
///
/// ```swift
/// let engine = GopherLLMEngine()
/// try await engine.load(modelAt: url)
/// for try await event in engine.stream([.user("Hello!")]) {
///     if case .delta(let text) = event { print(text, terminator: "") }
/// }
/// ```
public final class GopherLLMEngine: @unchecked Sendable {
    // The handle is an immutable integer; the Go side synchronizes all
    // access to the engine behind it.
    private let handle: gopherllm_engine

    private static let queue = DispatchQueue(label: "GopherLLM", qos: .userInitiated, attributes: .concurrent)

    public init() {
        handle = gopherllm_engine_new()
    }

    deinit {
        gopherllm_engine_free(handle)
    }

    // MARK: Library

    /// The GopherLLM library version.
    public static var version: String { takeString(gopherllm_version()) ?? "" }

    public static func runtimeInfo() throws -> RuntimeInfo {
        try JSON.decode(RuntimeInfo.self, from: takeString(gopherllm_runtime_info_json()) ?? "{}")
    }

    /// Reads only the GGUF header (fast for any file size), e.g. to check
    /// that a model is supported and fits in memory before loading it.
    public static func inspectModel(at url: URL) async throws -> ModelInfo {
        let path = url.path
        return try await Self.perform {
            let json = try check { gopherllm_inspect_model(path, $0) }
            return try JSON.decode(ModelInfo.self, from: json)
        }
    }

    // MARK: Model lifecycle

    /// Loads a GGUF model. Fails if a model is already loaded.
    public func load(modelAt url: URL, options: LoadOptions = LoadOptions()) async throws {
        let path = url.path
        let optionsJSON = try JSON.encode(options)
        try await Self.perform { [self] in
            if let message = takeString(gopherllm_load(handle, path, optionsJSON)) {
                throw GopherLLMError(message)
            }
        }
    }

    /// Stops any running request and releases the model's memory. The engine
    /// stays usable; call this on a memory warning or when backgrounded.
    public func unload() async throws {
        try await Self.perform { [self] in
            if let message = takeString(gopherllm_unload(handle)) {
                throw GopherLLMError(message)
            }
        }
    }

    public var isLoaded: Bool { gopherllm_is_loaded(handle) != 0 }

    /// The loaded model's name, or "".
    public var modelName: String { takeString(gopherllm_model_name(handle)) ?? "" }

    /// The loaded model's details, or `nil` when none is loaded.
    public func info() throws -> ModelInfo? {
        let json = takeString(gopherllm_info_json(handle)) ?? "{}"
        return json == "{}" ? nil : try JSON.decode(ModelInfo.self, from: json)
    }

    /// The number of tokens `text` encodes to with the loaded model.
    public func countTokens(_ text: String) async throws -> Int {
        try await Self.perform { [self] in
            var error: UnsafeMutablePointer<CChar>?
            let count = gopherllm_count_tokens(handle, text, &error)
            if let message = takeString(error) { throw GopherLLMError(message) }
            return Int(count)
        }
    }

    /// Stops the running request, if any.
    public func cancel() {
        gopherllm_cancel(handle)
    }

    // MARK: Generation

    /// Answers a single prompt and returns the generated text.
    public func generate(_ prompt: String, options: GenerationOptions = GenerationOptions()) async throws -> String {
        let optionsJSON = try JSON.encode(options)
        return try await cancellable { [self] in
            try check { gopherllm_generate(handle, prompt, optionsJSON, $0) }
        }
    }

    /// Continues a conversation and returns the full result.
    public func chat(_ messages: [ChatMessage], options: GenerationOptions = GenerationOptions()) async throws -> GenerationResult {
        let messagesJSON = try JSON.encode(messages)
        let optionsJSON = try JSON.encode(options)
        let json = try await cancellable { [self] in
            try check { gopherllm_chat(handle, messagesJSON, optionsJSON, $0) }
        }
        return try JSON.decode(GenerationResult.self, from: json)
    }

    /// Continues a conversation, delivering text as it is generated and the
    /// full result last. Ending the iteration early stops generation.
    public func stream(_ messages: [ChatMessage], options: GenerationOptions = GenerationOptions()) -> AsyncThrowingStream<StreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let messagesJSON: String
            let optionsJSON: String
            do {
                messagesJSON = try JSON.encode(messages)
                optionsJSON = try JSON.encode(options)
            } catch {
                continuation.finish(throwing: error)
                return
            }
            continuation.onTermination = { [handle] termination in
                if case .cancelled = termination { gopherllm_cancel(handle) }
            }
            let sink = StreamSink(continuation)
            Self.queue.async { [self] in
                // The callbacks all run before gopherllm_chat_stream returns,
                // so the sink only has to outlive this call.
                let failure = withExtendedLifetime(sink) {
                    gopherllm_chat_stream(handle, messagesJSON, optionsJSON,
                                          streamDelta, streamComplete, streamError,
                                          Unmanaged.passUnretained(sink).toOpaque())
                }
                if let message = takeString(failure) {
                    continuation.finish(throwing: GopherLLMError(message))
                }
            }
        }
    }

    /// Runs blocking model work off the caller's thread; cancelling the
    /// awaiting task cancels the engine's running request.
    private func cancellable<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
        try await withTaskCancellationHandler {
            try await Self.perform(work)
        } onCancel: { [handle] in
            gopherllm_cancel(handle)
        }
    }

    private static func perform<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                continuation.resume(with: Result { try work() })
            }
        }
    }
}

// MARK: C bridging

/// Copies and frees a string returned by the C API.
private func takeString(_ pointer: UnsafeMutablePointer<CChar>?) -> String? {
    guard let pointer else { return nil }
    defer { gopherllm_free_string(pointer) }
    return String(cString: pointer)
}

/// Applies the C API's error_out convention.
private func check(_ call: (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> UnsafeMutablePointer<CChar>?) throws -> String {
    var error: UnsafeMutablePointer<CChar>?
    let result = takeString(call(&error))
    if let message = takeString(error) { throw GopherLLMError(message) }
    guard let result else { throw GopherLLMError("GopherLLM returned no result") }
    return result
}

private final class StreamSink: Sendable {
    let continuation: AsyncThrowingStream<StreamEvent, Error>.Continuation

    init(_ continuation: AsyncThrowingStream<StreamEvent, Error>.Continuation) {
        self.continuation = continuation
    }

    static func from(_ userdata: UnsafeMutableRawPointer?) -> StreamSink {
        Unmanaged<StreamSink>.fromOpaque(userdata!).takeUnretainedValue()
    }
}

private let streamDelta: gopherllm_delta_cb = { delta, userdata in
    StreamSink.from(userdata).continuation.yield(.delta(String(cString: delta)))
}

private let streamComplete: gopherllm_complete_cb = { resultJSON, userdata in
    let continuation = StreamSink.from(userdata).continuation
    do {
        continuation.yield(.completed(try JSON.decode(GenerationResult.self, from: String(cString: resultJSON))))
        continuation.finish()
    } catch {
        continuation.finish(throwing: error)
    }
}

private let streamError: gopherllm_error_cb = { message, userdata in
    StreamSink.from(userdata).continuation.finish(throwing: GopherLLMError(String(cString: message)))
}
