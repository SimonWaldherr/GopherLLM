import SwiftUI

struct ContentView: View {
    @EnvironmentObject var vm: LLMViewModel
    @State private var importing = false
    private var threadLabel: String { vm.threads == 0 ? "Auto" : String(vm.threads) }
    var body: some View {
        NavigationStack { Form {
            Section("Model") {
                Picker("Local GGUF", selection: $vm.selected) { Text("Choose after import").tag(StoredModel?.none); ForEach(vm.models) { Text($0.name).tag(Optional($0)) } }
                Button("Import GGUF") { importing = true }
                HStack { Button("Load") { vm.load() }.disabled(vm.selected == nil || vm.status == .loading); Button("Unload", action: vm.unload).disabled(vm.status == .noModel) }
                Text("Status: \(vm.status.rawValue)")
            }
            Section("Chat") {
                ScrollView { Text(vm.transcript.isEmpty ? "Local, offline inference." : vm.transcript).frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled) }.frame(minHeight: 220)
                TextEditor(text: $vm.prompt).frame(height: 80)
                HStack { Button("Send", action: vm.send).disabled(vm.status != .ready); Button("Stop", action: vm.stop).disabled(vm.status != .generating); Button("Clear", action: vm.clear) }
            }
            Section("Settings") {
                Stepper("Max tokens: \(vm.maxTokens)", value: $vm.maxTokens, in: 1...2048)
                Slider(value: $vm.temperature, in: 0...2) { Text("Temperature \(vm.temperature, format: .number.precision(.fractionLength(2)))") }
                Slider(value: $vm.topP, in: 0.05...1) { Text("Top P") }
                Stepper("Top K: \(vm.topK)", value: $vm.topK, in: 0...200)
                Stepper("Threads: \(threadLabel)", value: $vm.threads, in: 0...12)
                Toggle("Use Metal GPU (eligible Qwen/Ministral)", isOn: $vm.useMetal)
                Slider(value: $vm.repeatPenalty, in: 0.5...2) { Text("Repeat penalty") }
            }
            if !vm.errorMessage.isEmpty { Section("Error") { Text(vm.errorMessage).foregroundStyle(.red) } }
        }.navigationTitle("GopherLLM").fileImporter(isPresented: $importing, allowedContentTypes: [.init(filenameExtension: "gguf")!], allowsMultipleSelection: false, onCompletion: vm.importModel) }
    }
}
