import Foundation
import GopherLLM

@MainActor
final class LLMViewModel: ObservableObject {
    enum Status: String {
        case noModel = "No model", copying = "Copying", loading = "Loading", ready = "Ready", generating = "Generating", error = "Error"
    }

    @Published var status: Status = .noModel
    @Published var models = ModelStore.listModels()
    @Published var selected: StoredModel? { didSet { inspectSelected() } }
    @Published var selectedInfo: ModelInfo?
    @Published var messages: [ChatMessage] = []
    @Published var streamingReply = ""
    @Published var lastStats = ""
    @Published var prompt = ""
    @Published var errorMessage = ""
    @Published var maxTokens = 256
    @Published var temperature = 0.7
    @Published var topP = 0.9
    @Published var topK = 40
    @Published var repeatPenalty = 1.1
    @Published var threads = 0
    @Published var useMetal = false

    let metalAvailable = (try? GopherLLMEngine.runtimeInfo().metalAvailable) ?? false
    private let engine = GopherLLMEngine()
    private var generation: Task<Void, Never>?

    func importModel(_ result: Result<[URL], Error>) {
        guard case let .success(urls) = result, let url = urls.first else { return }
        status = .copying
        Task {
            do {
                // Copying a multi-GB file must not block the main thread.
                let model = try await Task.detached { try ModelStore.importModel(url) }.value
                models = ModelStore.listModels()
                selected = model
                status = .noModel
            } catch {
                fail(error)
            }
        }
    }

    func load() {
        guard let selected else { return }
        errorMessage = ""
        status = .loading
        let options = LoadOptions(threads: threads, metal: useMetal)
        Task {
            do {
                if engine.isLoaded { try await engine.unload() }
                try await engine.load(modelAt: selected.url, options: options)
                status = .ready
            } catch {
                fail(error)
            }
        }
    }

    func unload() {
        generation?.cancel()
        Task {
            try? await engine.unload()
            status = .noModel
        }
    }

    /// iOS terminates apps that keep growing under memory pressure; releasing
    /// the model is the one large allocation the app can give back.
    func handleMemoryWarning() {
        guard engine.isLoaded else { return }
        unload()
        errorMessage = "The model was unloaded because the system is low on memory."
    }

    func send() {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, status == .ready else { return }
        prompt = ""
        errorMessage = ""
        messages.append(.user(text))
        status = .generating
        let history = messages
        // "recent" drops the oldest turns once the chat outgrows the context.
        let options = GenerationOptions(maxTokens: maxTokens, temperature: Float(temperature), topP: Float(topP),
                                        topK: topK, repeatPenalty: Float(repeatPenalty), contextWindowMode: .recent)
        generation = Task {
            do {
                for try await event in engine.stream(history, options: options) {
                    switch event {
                    case .delta(let piece):
                        streamingReply += piece
                    case .completed(let result):
                        lastStats = String(format: "%d tokens, %.1f tokens/s", result.generatedTokens, result.tokensPerSecond)
                    }
                }
            } catch {
                if !Task.isCancelled { errorMessage = error.localizedDescription }
            }
            if !streamingReply.isEmpty { messages.append(.assistant(streamingReply)) }
            streamingReply = ""
            status = .ready
        }
    }

    func stop() { generation?.cancel() }

    func clear() {
        messages.removeAll()
        lastStats = ""
    }

    private func inspectSelected() {
        selectedInfo = nil
        guard let url = selected?.url else { return }
        Task { selectedInfo = try? await GopherLLMEngine.inspectModel(at: url) }
    }

    private func fail(_ error: Error) {
        errorMessage = error.localizedDescription
        status = .error
    }
}
