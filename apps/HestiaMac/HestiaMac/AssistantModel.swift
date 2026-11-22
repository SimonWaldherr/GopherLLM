import AppKit
import AVFoundation
import Foundation
import GopherLLM

@MainActor
final class AssistantModel: ObservableObject {
    @Published var input = ""
    @Published var activation = UserDefaults.standard.string(forKey: "wake") ?? "Computer"
    @Published var backend = SpeechBackend.phonon
    @Published var reference = ""
    @Published var artifact = ""
    @Published var configuration = UserDefaults.standard.string(forKey: "configuration") ?? "{}"
    @Published var homeToken = ""
    @Published var hubToken = ""
    @Published private(set) var turns: [HestiaTurn] = []
    @Published private(set) var status = "Hestia wird gestartet …"
    @Published private(set) var error = ""
    @Published private(set) var modelName = ""
    @Published private(set) var captureMode: Microphone.Mode?
    @Published private(set) var busy = false
    @Published private(set) var ready = false
    @Published private(set) var downloading = false
    @Published private(set) var downloadProgress = 0.0

    private var host: NativeHestia?
    private var speech: GopherLLMSpeech?
    private var capture: Microphone?
    private var captureID: UUID?
    private var task: Task<Void, Never>?

    var canRecord: Bool { speech != nil && !busy }
    var backends: [SpeechBackend] { SpeechBackend.allCases.filter(\.isAvailable) }

    func selectBackend(_ kind: SpeechBackend) {
        switch kind {
        case .phonon: reference = ""; artifact = ""
        case .voxtral, .whisper: reference = ""; artifact = ""
        }
    }

    private func support() throws -> URL {
        try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("Hestia", isDirectory: true)
    }

    func start() {
        guard host == nil, task == nil else { return }
        homeToken = Credentials.read("home-assistant")
        hubToken = Credentials.read("hugging-face")
        perform("Hestia wird gestartet …") { [self] in
            host = try await NativeHestia.open(configuration: configuration, directory: support(), token: homeToken)
            ready = true
            status = "Bereit"
            if let path = UserDefaults.standard.string(forKey: "speechPath"),
               let saved = UserDefaults.standard.string(forKey: "speechBackend"),
               let kind = SpeechBackend(rawValue: saved), kind.isAvailable,
               FileManager.default.fileExists(atPath: path) {
                backend = kind
                try await loadSpeech(URL(fileURLWithPath: path), backend: kind)
            }
        }
    }

    func saveSettings() {
        stopCapture()
        perform("Einstellungen werden angewendet …") { [self] in
            // Close the previous database before opening the same directory.
            host = nil
            ready = false
            do {
                host = try await NativeHestia.open(configuration: configuration, directory: support(), token: homeToken)
                ready = true
                turns = []
                try Credentials.save(homeToken, account: "home-assistant")
                try Credentials.save(hubToken, account: "hugging-face")
                UserDefaults.standard.set(configuration, forKey: "configuration")
                UserDefaults.standard.set(activation, forKey: "wake")
                status = "Einstellungen gespeichert"
            } catch {
                if host == nil {
                    let previous = UserDefaults.standard.string(forKey: "configuration") ?? "{}"
                    let previousToken = Credentials.read("home-assistant")
                    host = try? await NativeHestia.open(configuration: previous, directory: support(), token: previousToken)
                    ready = host != nil
                }
                throw error
            }
        }
    }

    func send() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, let host, !busy else { return }
        input = ""
        perform("Hestia verarbeitet den Befehl …") { [self] in
            append(try await host.command(text))
        }
    }

    func confirm(_ turn: HestiaTurn, device: String) {
        guard let host else { return }
        perform("Bestätigter Befehl wird ausgeführt …") { [self] in
            append(try await host.confirm(turn.id, device: device))
        }
    }

    private func append(_ turn: HestiaTurn) {
        if let i = turns.firstIndex(where: { $0.id == turn.id }) { turns[i] = turn }
        else { turns.append(turn) }
        status = "Bereit"
    }

    func capture(_ mode: Microphone.Mode) {
        guard canRecord, captureMode == nil, ready else { return }
        perform("Mikrofon wird gestartet …") { [self] in
            let allowed = await AVCaptureDevice.requestAccess(for: .audio)
            guard allowed else { throw GopherLLMError("Erlaube Hestia den Mikrofonzugriff in den Systemeinstellungen.") }
            try Task.checkCancellation()
            guard NSApplication.shared.isActive else { throw GopherLLMError("Öffne das Hestia-Fenster für die Aufnahme.") }
            let word = activation.trimmingCharacters(in: .whitespacesAndNewlines)
            if mode == .wake && word.isEmpty { throw GopherLLMError("Gib zuerst ein Aktivierungswort ein.") }
            let id = UUID()
            capture = try Microphone(mode: mode) { [weak self] clip in
                Task { @MainActor [weak self] in
                    guard let self, self.captureID == id, !self.busy else { return }
                    if mode == .manual { self.stopCapture() }
                    self.transcribe(clip, activationPhrase: mode == .wake ? word : nil)
                }
            }
            captureID = id
            captureMode = mode
            status = mode == .manual ? "Aufnahme läuft" : "Hört auf „\(word)“"
        }
    }

    func finishRecording() {
        let clip = stopCapture()
        if !clip.isEmpty { transcribe(clip, activationPhrase: nil) }
    }

    @discardableResult
    func stopCapture() -> [Float] {
        captureID = nil
        captureMode = nil
        let clip = capture?.stop() ?? []
        capture = nil
        if !busy { status = "Bereit" }
        return clip
    }

    private func transcribe(_ clip: [Float], activationPhrase: String?) {
        guard let speech, let host else { return }
        perform("Sprache wird lokal erkannt …") { [self] in
            let transcript = try await speech.transcribe(clip, activationPhrase: activationPhrase)
            try Task.checkCancellation()
            if !transcript.text.isEmpty && transcript.matched != false {
                append(try await host.command(transcript.text))
            }
            status = captureMode == .wake ? "Hört auf „\(activationPhrase ?? activation)“" : "Bereit"
        }
    }

    func downloadSpeech() {
        let ref = reference.trimmingCharacters(in: .whitespacesAndNewlines)
        let file = artifact.trimmingCharacters(in: .whitespacesAndNewlines)
        let kind = backend
        perform("Modell wird gesucht …") { [self] in
            downloading = true
            downloadProgress = 0
            defer { downloading = false }
            let cache = try support().appendingPathComponent("Models", isDirectory: true)
            let hub = GopherLLMHub(cacheDirectory: cache, token: hubToken.isEmpty ? nil : hubToken)
            let files: [String]?
            if kind == .phonon {
                let manifest = try await hub.inspect(ref)
                let paths = Set(manifest.files.map(\.path))
                let candidates = file.isEmpty ? paths.sorted().filter {
                    $0 == "model.fermion" || $0.hasSuffix("/model.fermion")
                } : [file]
                guard let pair = candidates.lazy.compactMap({ weight -> [String]? in
                    guard weight.hasSuffix(".fermion"), paths.contains(weight) else { return nil }
                    let prefix = weight.lastIndex(of: "/").map { String(weight[...$0]) } ?? ""
                    let config = prefix + "config.json"
                    return paths.contains(config) ? [weight, config] : nil
                }).first else {
                    throw GopherLLMError("Der Hub muss die .fermion-Datei und config.json im selben Ordner anbieten. Alternativ kannst du den entpackten Phonon-Ordner importieren.")
                }
                files = pair
            } else if kind == .whisper {
                guard !file.isEmpty else { throw GopherLLMError("Gib den Namen einer Whisper-Datei an, etwa ggml-base.bin.") }
                files = [file]
            } else {
                files = file.isEmpty ? nil : [file]
            }
            status = "Sprachmodell wird heruntergeladen …"
            let urls = try await hub.download(ref, files: files) { [weak self] event in
                guard event.totalBytes > 0 else { return }
                Task { @MainActor [weak self] in
                    self?.downloadProgress = min(1, Double(event.completedBytes) / Double(event.totalBytes))
                }
            }
            guard let url = urls.first else { throw GopherLLMError("Keine Modelldatei gefunden.") }
            try await loadSpeech(url, backend: kind)
        }
    }

    func importSpeech(_ url: URL) {
        let kind = backend
        perform("Modell wird importiert …") { [self] in
            let dest = try support().appendingPathComponent("Imported/\(UUID().uuidString)", isDirectory: true)
            let local = try await Task.detached {
                let access = url.startAccessingSecurityScopedResource()
                defer { if access { url.stopAccessingSecurityScopedResource() } }
                let folder = try url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true
                let source = folder ? url.appendingPathComponent("model.fermion") : url
                try FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)
                do {
                    let target = dest.appendingPathComponent(source.lastPathComponent)
                    try FileManager.default.copyItem(at: source, to: target)
                    if folder {
                        try FileManager.default.copyItem(at: url.appendingPathComponent("config.json"), to: dest.appendingPathComponent("config.json"))
                    }
                    return target
                } catch {
                    try? FileManager.default.removeItem(at: dest)
                    throw error
                }
            }.value
            do { try await loadSpeech(local, backend: kind) }
            catch {
                try? FileManager.default.removeItem(at: dest)
                throw error
            }
        }
    }

    func importAssistantModel(_ url: URL, configKey: String) {
        perform("GGUF wird in den App-Ordner kopiert …") { [self] in
            guard url.pathExtension.lowercased() == "gguf" else { throw GopherLLMError("Wähle eine GGUF-Datei.") }
            guard var settings = try JSONSerialization.jsonObject(with: Data(configuration.utf8)) as? [String: Any] else {
                throw GopherLLMError("Die Hestia-Konfiguration muss ein JSON-Objekt sein.")
            }
            let folder = try support().appendingPathComponent("Imported/\(UUID().uuidString)", isDirectory: true)
            let local = try await Task.detached {
                let access = url.startAccessingSecurityScopedResource()
                defer { if access { url.stopAccessingSecurityScopedResource() } }
                let target = folder.appendingPathComponent(url.lastPathComponent)
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                do { try FileManager.default.copyItem(at: url, to: target) }
                catch { try? FileManager.default.removeItem(at: folder); throw error }
                return target
            }.value
            settings[configKey] = local.path
            let json = try JSONSerialization.data(withJSONObject: settings, options: [.prettyPrinted, .sortedKeys])
            configuration = String(decoding: json, as: UTF8.self)
            status = "Modell importiert. Wende die Einstellungen an, um es zu laden."
        }
    }

    func importDocument(_ url: URL) {
        guard let host else { return }
        perform("Dokument wird lokal indexiert …") { [self] in
            let text = try await Task.detached {
                let access = url.startAccessingSecurityScopedResource()
                defer { if access { url.stopAccessingSecurityScopedResource() } }
                let file = try FileHandle(forReadingFrom: url)
                defer { try? file.close() }
                let data = try file.read(upToCount: (1 << 20) + 1) ?? Data()
                guard data.count <= 1 << 20, let text = String(data: data, encoding: .utf8) else {
                    throw GopherLLMError("Wähle eine UTF-8-Textdatei mit höchstens 1 MiB.")
                }
                return text
            }.value
            try await host.addDocument(title: url.deletingPathExtension().lastPathComponent, text: text)
            status = "Dokument indexiert"
        }
    }

    private func loadSpeech(_ url: URL, backend: SpeechBackend) async throws {
        status = "Sprachmodell wird lokal geladen …"
        let model = try await GopherLLMSpeech.open(modelAt: url, options: .init(backend: backend))
        speech = model
        modelName = url.lastPathComponent
        UserDefaults.standard.set(url.path, forKey: "speechPath")
        UserDefaults.standard.set(backend.rawValue, forKey: "speechBackend")
        status = "Bereit"
    }

    func cancel() { task?.cancel(); stopCapture() }
    func report(_ error: Error) { self.error = error.localizedDescription }

    private func perform(_ message: String, work: @escaping @MainActor () async throws -> Void) {
        guard task == nil else { return }
        busy = true
        error = ""
        status = message
        task = Task {
            defer { task = nil; busy = false }
            do { try await work() }
            catch is CancellationError { status = "Abgebrochen" }
            catch { report(error); status = "Bereit" }
        }
    }
}
