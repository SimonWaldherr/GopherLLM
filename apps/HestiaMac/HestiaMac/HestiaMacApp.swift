import AppKit
import SwiftUI
import UniformTypeIdentifiers
import GopherLLM

@main
struct HestiaMacApp: App {
    var body: some Scene {
        Window("Hestia", id: "assistant") {
            ContentView()
                .frame(minWidth: 570, minHeight: 460)
        }
        .defaultSize(width: 690, height: 580)
    }
}

private struct ContentView: View {
    @StateObject private var model = AssistantModel()
    @State private var importing: ImportKind?

    private enum ImportKind { case speech, document, embed, chat }

    var body: some View {
        TabView {
            conversation.tabItem { Label("Assistent", systemImage: "waveform") }
            settings.tabItem { Label("Einstellungen", systemImage: "gearshape") }
        }
        .padding(14)
        .task { model.start() }
        .onDisappear { model.cancel() }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didResignActiveNotification)) { _ in model.stopCapture() }
        .safeAreaInset(edge: .bottom) { statusBar }
        .fileImporter(isPresented: Binding(get: { importing != nil }, set: { if !$0 { importing = nil } }),
                      allowedContentTypes: importing == .document ? [.plainText] : [.data, .folder]) { [kind = importing] result in
            switch result {
            case .success(let url):
                switch kind {
                case .document: model.importDocument(url)
                case .embed: model.importAssistantModel(url, configKey: "embed_model_path")
                case .chat: model.importAssistantModel(url, configKey: "chat_model_path")
                case .speech: model.importSpeech(url)
                case nil: break
                }
            case .failure(let error): model.report(error)
            }
        }
    }

    private var conversation: some View {
        VStack(alignment: .leading, spacing: 12) {
            if model.turns.isEmpty {
                ContentUnavailableView("Hestia", systemImage: "waveform", description: Text("Schalte die Stehlampe im Wohnzimmer ein."))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollViewReader { scroll in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 14) {
                            ForEach(model.turns) { turn in
                                turnView(turn).id(turn.id)
                            }
                        }
                        .padding(.vertical, 6)
                    }
                    .onChange(of: model.turns.count) { _, _ in
                        if let last = model.turns.last { scroll.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }
            Divider()
            HStack {
                TextField("Nachricht oder Befehl", text: $model.input)
                    .onSubmit(model.send)
                    .disabled(!model.ready || model.busy || model.captureMode != nil)
                Button("Senden", action: model.send)
                    .disabled(!model.ready || model.busy || model.captureMode != nil || model.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            HStack {
                Button {
                    if model.captureMode == .manual { model.finishRecording() }
                    else { model.capture(.manual) }
                } label: {
                    Label(model.captureMode == .manual ? "Aufnahme abschließen" : "Aufnehmen", systemImage: model.captureMode == .manual ? "stop.circle.fill" : "mic")
                }
                .keyboardShortcut("r", modifiers: [.command, .shift])
                .disabled(!model.canRecord || model.captureMode == .wake)
                Button {
                    if model.captureMode == .wake { model.cancel() }
                    else { model.capture(.wake) }
                } label: {
                    Label(model.captureMode == .wake ? "Zuhören stoppen" : "Zuhören", systemImage: model.captureMode == .wake ? "stop.circle" : "ear")
                }
                .disabled(model.captureMode != .wake && (!model.canRecord || model.captureMode == .manual))
                Spacer()
                Button("Dokument importieren …") { importing = .document }
                    .disabled(!model.ready || model.busy || model.captureMode != nil)
            }
        }
        .padding(12)
    }

    private func turnView(_ turn: HestiaTurn) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(turn.InputText).font(.headline).textSelection(.enabled)
            Text(turn.Result).textSelection(.enabled)
            if let sources = turn.sources {
                ForEach(Array(sources.enumerated()), id: \.offset) { _, source in
                    if let url = URL(string: source.url), ["http", "https"].contains(url.scheme?.lowercased() ?? "") {
                        Link(source.title, destination: url)
                        if !source.snippet.isEmpty { Text(source.snippet).font(.callout) }
                    }
                }
            }
            if turn.Status == "clarifying" {
                if turn.Proposal?.Scope == "group" {
                    Button("Gruppe bestätigen") { model.confirm(turn, device: "") }.disabled(model.busy)
                } else {
                    ForEach(turn.Candidates ?? []) { device in
                        Button("\(device.Name) · \(device.Room)") { model.confirm(turn, device: device.id) }
                            .disabled(model.busy)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(.quaternary.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
    }

    private var settings: some View {
        Form {
            Section("Sprache") {
                Picker("Sprachmodell", selection: $model.backend) {
                    ForEach(model.backends, id: \.self) { backend in Text(backend.rawValue.capitalized).tag(backend) }
                }
                .onChange(of: model.backend) { _, kind in model.selectBackend(kind) }
                TextField("Aktivierungswort", text: $model.activation)
                TextField("Hugging-Face-Repository", text: $model.reference)
                TextField(model.backend == .whisper ? "Dateiname (Whisper GGML)" : "Dateiname (optional)", text: $model.artifact)
                HStack {
                    Button("Von Hugging Face laden", action: model.downloadSpeech)
                    Button("Modell importieren …") { importing = .speech }
                    if !model.modelName.isEmpty { Text(model.modelName).foregroundStyle(.secondary) }
                }
                if model.downloading { ProgressView(value: model.downloadProgress) }
            }
            Section("Assistent") {
                Text("Optionales Hestia-JSON für Wikipedia, Websuche, Home Assistant und lokale Modelle.")
                    .foregroundStyle(.secondary)
                TextEditor(text: $model.configuration).font(.system(.body, design: .monospaced)).frame(minHeight: 95)
                    .accessibilityLabel("Hestia-Konfiguration")
                SecureField("Home-Assistant-Token", text: $model.homeToken)
                SecureField("Hugging-Face-Token", text: $model.hubToken)
                HStack {
                    Button("Embedding-Modell importieren …") { importing = .embed }
                    Button("Chatmodell importieren …") { importing = .chat }
                }
                Button("Einstellungen anwenden", action: model.saveSettings)
            }
            Section("Datenschutz") {
                Text("Aufnahmen werden lokal verarbeitet. Hestia aktiviert das Mikrofon erst auf deine Anforderung und stoppt es beim Wechsel zu einer anderen App.")
                Text("Hugging Face erhält Modellanfragen. Externe Suchdienste und Home Assistant erhalten Anfragen nur bei konfiguriertem Zugriff.")
                DisclosureGroup("Datenschutzerklärung") {
                    if let url = Bundle.main.url(forResource: "PRIVACY", withExtension: "md"),
                       let policy = try? String(contentsOf: url, encoding: .utf8) {
                        Text(policy).textSelection(.enabled)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .disabled(model.busy || model.captureMode != nil)
    }

    private var statusBar: some View {
        VStack(alignment: .leading, spacing: 5) {
            if !model.error.isEmpty { Text(model.error).foregroundStyle(.red).textSelection(.enabled) }
            HStack {
                if model.captureMode != nil { Image(systemName: "mic.fill").foregroundStyle(.red) }
                Text(model.status)
                Spacer()
                if model.busy || model.captureMode != nil { Button("Abbrechen", action: model.cancel) }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.regularMaterial)
    }
}
