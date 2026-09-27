import GopherLLM
import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var vm: LLMViewModel
    @State private var importing = false

    var body: some View {
        NavigationStack {
            Form {
                modelSection
                chatSection
                settingsSection
                if !vm.errorMessage.isEmpty {
                    Section("Error") { Text(vm.errorMessage).foregroundStyle(.red) }
                }
            }
            .navigationTitle("GopherLLM")
            .fileImporter(isPresented: $importing, allowedContentTypes: [UTType(filenameExtension: "gguf") ?? .data],
                          allowsMultipleSelection: false, onCompletion: vm.importModel)
            .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
                vm.handleMemoryWarning()
            }
        }
    }

    private var modelSection: some View {
        Section("Model") {
            Picker("Local GGUF", selection: $vm.selected) {
                Text("Choose after import").tag(StoredModel?.none)
                ForEach(vm.models) { Text($0.name).tag(Optional($0)) }
            }
            if let info = vm.selectedInfo {
                LabeledContent("Architecture", value: info.supported ? info.architecture : "\(info.architecture) (unsupported)")
                LabeledContent("Weights", value: "\(info.quantization), \(ByteCountFormatter.string(fromByteCount: info.tensorBytes, countStyle: .memory))")
                LabeledContent("Context", value: "\(info.contextLength) tokens")
            }
            Button("Import GGUF") { importing = true }
            HStack {
                Button("Load", action: vm.load)
                    .disabled(vm.selected == nil || vm.status == .loading || vm.status == .generating)
                Spacer()
                Button("Unload", action: vm.unload).disabled(vm.status == .noModel)
            }
            .buttonStyle(.borderless)
            LabeledContent("Status", value: vm.status.rawValue)
        }
    }

    private var chatSection: some View {
        Section("Chat") {
            if vm.messages.isEmpty && vm.streamingReply.isEmpty {
                Text("Local, offline inference.").foregroundStyle(.secondary)
            }
            ForEach(Array(vm.messages.enumerated()), id: \.offset) { _, message in
                MessageRow(role: message.role, text: message.content)
            }
            if vm.status == .generating {
                MessageRow(role: .assistant, text: vm.streamingReply.isEmpty ? "…" : vm.streamingReply)
            }
            if !vm.lastStats.isEmpty {
                Text(vm.lastStats).font(.caption).foregroundStyle(.secondary)
            }
            TextField("Message", text: $vm.prompt, axis: .vertical).lineLimit(1...5)
            HStack {
                Button("Send", action: vm.send).disabled(vm.status != .ready)
                Spacer()
                Button("Stop", action: vm.stop).disabled(vm.status != .generating)
                Spacer()
                Button("Clear", action: vm.clear).disabled(vm.status == .generating)
            }
            .buttonStyle(.borderless)
        }
    }

    private var settingsSection: some View {
        Section("Settings") {
            Stepper("Max tokens: \(vm.maxTokens)", value: $vm.maxTokens, in: 16...4096, step: 16)
            VStack(alignment: .leading) {
                Text("Temperature \(vm.temperature, format: .number.precision(.fractionLength(2)))")
                Slider(value: $vm.temperature, in: 0...2)
            }
            VStack(alignment: .leading) {
                Text("Top P \(vm.topP, format: .number.precision(.fractionLength(2)))")
                Slider(value: $vm.topP, in: 0.05...1)
            }
            Stepper("Top K: \(vm.topK)", value: $vm.topK, in: 0...200)
            VStack(alignment: .leading) {
                Text("Repeat penalty \(vm.repeatPenalty, format: .number.precision(.fractionLength(2)))")
                Slider(value: $vm.repeatPenalty, in: 1...2)
            }
            Stepper("Threads: \(vm.threads == 0 ? "Auto" : String(vm.threads))", value: $vm.threads, in: 0...12)
            Toggle("Use Metal GPU (applies on next load)", isOn: $vm.useMetal).disabled(!vm.metalAvailable)
        }
    }
}

private struct MessageRow: View {
    let role: ChatMessage.Role
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(role == .user ? "You" : "Assistant").font(.caption).foregroundStyle(.secondary)
            Text(text).textSelection(.enabled)
        }
    }
}
