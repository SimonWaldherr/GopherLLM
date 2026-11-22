import Foundation
import GopherLLMCore

public struct HubRepository: Decodable, Sendable, Equatable, Identifiable {
    public let id: String
    public let pipelineTag: String?
    public let libraryName: String?
    public let tags: [String]?
    public let downloads: Int64
    public let gated: Bool
    public let gguf: Bool
}

public struct HubManifest: Decodable, Sendable, Equatable {
    public struct File: Decodable, Sendable, Equatable {
        public let path: String
        public let sizeBytes: Int64
        public let format: String
        public let role: String
    }

    public let repository: String
    public let revision: String
    public let commit: String?
    public let files: [File]
}

public struct HubProgress: Sendable, Equatable {
    public let file: String
    public let completedBytes: Int64
    public let totalBytes: Int64
    public let done: Bool
    public let failure: String?
}

/// Downloads model data from Hugging Face into a host-selected cache. An
/// app can use a folder inside its sandbox and pass a token from its Keychain.
/// Instances never read HF_TOKEN or HF_ENDPOINT from the process environment.
public struct GopherLLMHub: Sendable {
    public let cacheDirectory: URL
    public let offline: Bool
    private let token: String?

    public init(cacheDirectory: URL, token: String? = nil, offline: Bool = false) {
        self.cacheDirectory = cacheDirectory
        self.token = token
        self.offline = offline
    }

    public func search(_ query: String, pipelineTag: String? = nil, format: String? = nil, limit: Int? = nil) async throws -> [HubRepository] {
        let json = try options(pipelineTag: pipelineTag, format: format, limit: limit)
        return try await NativeOperation.run { operation in
            let result = try check { gopherllm_hub_search(query, json, operation, $0) }
            return try JSON.decode([HubRepository].self, from: result)
        }
    }

    /// Lists repository artifacts without downloading weights.
    public func inspect(_ reference: String) async throws -> HubManifest {
        let json = try options()
        return try await NativeOperation.run { operation in
            let result = try check { gopherllm_hub_inspect(reference, json, operation, $0) }
            return try JSON.decode(HubManifest.self, from: result)
        }
    }

    /// Resolves a GGUF selector including all shards, or downloads exact
    /// selected model files. Callbacks may run concurrently across files.
    public func download(_ reference: String, files: [String]? = nil,
                         onProgress: (@Sendable (HubProgress) -> Void)? = nil) async throws -> [URL] {
        let json = try options()
        let filesJSON = try files.map { try JSON.encode($0) }
        let sink = onProgress.map(HubProgressSink.init)
        return try await NativeOperation.run { operation in
            let result = try withExtendedLifetime(sink) {
                let data = sink.map { Unmanaged.passUnretained($0).toOpaque() }
                return try withOptionalCString(filesJSON) { selection in
                    try check { gopherllm_hub_download(reference, selection, json, operation, data == nil ? nil : hubProgress, data, $0) }
                }
            }
            return try JSON.decode([String].self, from: result).map { URL(fileURLWithPath: $0) }
        }
    }

    private struct Options: Encodable {
        let cacheDir: String
        let token: String?
        let offline: Bool
        let pipelineTag: String?
        let format: String?
        let limit: Int?
    }

    private func options(pipelineTag: String? = nil, format: String? = nil, limit: Int? = nil) throws -> String {
        guard cacheDirectory.isFileURL else { throw GopherLLMError("Hub cache requires a local directory") }
        return try JSON.encode(Options(cacheDir: cacheDirectory.path, token: token, offline: offline,
                                       pipelineTag: pipelineTag, format: format, limit: limit))
    }
}

private final class HubProgressSink: Sendable {
    let callback: @Sendable (HubProgress) -> Void

    init(_ callback: @escaping @Sendable (HubProgress) -> Void) { self.callback = callback }
}

private let hubProgress: gopherllm_hub_progress_cb = { file, completed, total, done, failure, data in
    guard let data else { return }
    let sink = Unmanaged<HubProgressSink>.fromOpaque(data).takeUnretainedValue()
    sink.callback(HubProgress(file: String(cString: file), completedBytes: completed, totalBytes: total,
                              done: done != 0, failure: failure.map { String(cString: $0) }))
}
