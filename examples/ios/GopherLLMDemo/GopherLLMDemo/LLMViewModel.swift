import Foundation
import SwiftUI
import GopherLLM

@MainActor
final class LLMViewModel: ObservableObject {
    enum Status: String { case noModel = "No model", copying = "Copying", loading = "Loading", ready = "Ready", generating = "Generating", error = "Error" }
    @Published var status: Status = .noModel
    @Published var models = ModelStore.listModels()
    @Published var selected: StoredModel?
    @Published var transcript = ""
    @Published var prompt = ""
    @Published var errorMessage = ""
    @Published var maxTokens = 128
    @Published var temperature = 0.7
    @Published var topP = 0.9
    @Published var topK = 40
    @Published var repeatPenalty = 1.1
    @Published var threads = 0
    @Published var useMetal = false
    private let engine = MobileNewEngine()!
    private var task: Task<Void, Never>?

    func importModel(_ result: Result<[URL], Error>) {
        guard case let .success(urls) = result, let url = urls.first else { return }
        status = .copying
        Task.detached { [weak self] in
            do { let model = try ModelStore.importModel(url); await MainActor.run { self?.models = ModelStore.listModels(); self?.selected = model; self?.status = .noModel } }
            catch { await MainActor.run { self?.fail(error) } }
        }
    }
    func load() { guard let selected else { return }; status = .loading; let path = selected.url.path
        let engine = engine; let options = loadJSON()
        task = Task.detached { [weak self] in do { try engine.load(path, optionsJSON: options); await MainActor.run { self?.status = .ready } } catch { await MainActor.run { self?.fail(error) } } }
    }
    func unload() { engine.cancel(); task?.cancel(); Task.detached { [weak self] in _ = try? self?.engine.unload(); await MainActor.run { self?.status = .noModel } } }
    func send() { guard !prompt.isEmpty, status == .ready else { return }; let input = prompt; prompt = ""; transcript += "\n\nYou: \(input)\nAssistant: "; status = .generating
        let sink = DemoSink { [weak self] delta in Task { @MainActor in self?.transcript += delta } } complete: { [weak self] _ in Task { @MainActor in self?.status = .ready } } failure: { [weak self] message in Task { @MainActor in self?.failMessage(message) } }
        let engine = engine; let options = generationJSON()
        task = Task.detached { _ = try? engine.generateStream(input, optionsJSON: options, sink: sink) }
    }
    func stop() { engine.cancel() }
    func clear() { transcript = "" }
    private func loadJSON() -> String { json(["threads": threads, "prepare_quantized": false, "out_of_core": false, "prefault": "none", "metal": useMetal]) }
    private func generationJSON() -> String { json(["max_tokens": maxTokens, "temperature": temperature, "top_p": topP, "top_k": topK, "repeat_penalty": repeatPenalty, "min_p": 0]) }
    private func json(_ value: [String: Any]) -> String { String(data: try! JSONSerialization.data(withJSONObject: value), encoding: .utf8)! }
    private func fail(_ error: Error) { failMessage(error.localizedDescription) }
    private func failMessage(_ message: String) { errorMessage = message; status = .error }
}

final class DemoSink: NSObject, MobileStreamSinkProtocol {
    let delta: (String) -> Void; let complete: (String) -> Void; let failure: (String) -> Void
    init(delta: @escaping (String) -> Void, complete: @escaping (String) -> Void, failure: @escaping (String) -> Void) { self.delta = delta; self.complete = complete; self.failure = failure }
    func onDelta(_ text: String?) { delta(text ?? "") }
    func onComplete(_ resultJSON: String?) { complete(resultJSON ?? "") }
    func onError(_ message: String?) { failure(message ?? "Unknown stream error") }
}
