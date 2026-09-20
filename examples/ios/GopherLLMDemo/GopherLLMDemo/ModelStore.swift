import Foundation

struct StoredModel: Identifiable, Hashable {
    let url: URL
    var id: URL { url }
    var name: String { url.lastPathComponent }
    var size: Int64 { (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) ?? 0 }
}

enum ModelStore {
    static var directory: URL {
        let base = try! FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let dir = base.appendingPathComponent("GopherLLM/Models", isDirectory: true)
        try! FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
    static func listModels() -> [StoredModel] {
        (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.fileSizeKey]))?.filter { $0.pathExtension.lowercased() == "gguf" }.map(StoredModel.init(url:)).sorted { $0.name < $1.name } ?? []
    }
    static func importModel(_ source: URL) throws -> StoredModel {
        guard source.pathExtension.lowercased() == "gguf" else { throw CocoaError(.fileReadUnsupportedScheme) }
        guard source.startAccessingSecurityScopedResource() else { throw CocoaError(.fileReadNoPermission) }
        defer { source.stopAccessingSecurityScopedResource() }
        let destination = directory.appendingPathComponent(source.lastPathComponent)
        if FileManager.default.fileExists(atPath: destination.path) { return StoredModel(url: destination) }
        // FileManager copies directly between file providers; do not materialize a multi-GB GGUF as Data.
        try FileManager.default.copyItem(at: source, to: destination)
        return StoredModel(url: destination)
    }
}
