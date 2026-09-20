import SwiftUI

@main
struct GopherLLMDemoApp: App {
    @StateObject private var viewModel = LLMViewModel()

    var body: some Scene {
        WindowGroup { ContentView().environmentObject(viewModel) }
    }
}
