import Foundation
@_spi(AssistantHost) import GopherLLM
import GopherLLMCore
import Security

struct HestiaTurn: Decodable, Identifiable, Sendable {
    struct Device: Decodable, Identifiable, Sendable {
        let id: String
        let Name: String
        let Room: String
        private enum CodingKeys: String, CodingKey { case id = "ID", Name, Room }
    }
    struct Proposal: Decodable, Sendable { let Scope: String }
    struct Source: Decodable, Sendable { let title: String; let snippet: String; let url: String }
    let id: String
    let Status: String
    let InputText: String
    let Result: String
    let Proposal: Proposal?
    let Candidates: [Device]?
    let sources: [Source]?
    private enum CodingKeys: String, CodingKey { case id = "ID", Status, InputText, Result, Proposal, Candidates, sources }
}

final class NativeHestia: Sendable {
    private let handle: hestia_runtime
    private let conversation = UUID().uuidString

    private init(_ handle: hestia_runtime) { self.handle = handle }
    deinit { hestia_close(handle) }

    static func open(configuration: String, directory: URL, token: String) async throws -> NativeHestia {
        try await NativeOperation.run { op in
            var error: UnsafeMutablePointer<CChar>?
            let handle = token.withCString { hestia_open(configuration, directory.path, $0, op, &error) }
            if let message = take(error) { throw GopherLLMError(message) }
            guard handle != 0 else { throw GopherLLMError("Hestia konnte nicht gestartet werden.") }
            return NativeHestia(handle)
        }
    }
    func command(_ text: String) async throws -> HestiaTurn {
        guard !text.contains("\0") else { throw GopherLLMError("Ungültiges Zeichen im Befehl.") }
        return try await NativeOperation.run { [self] op in
            let data = try checked { hestia_command(handle, conversation, text, op, $0) }
            return try JSONDecoder().decode(HestiaTurn.self, from: data)
        }
    }
    func confirm(_ turn: String, device: String) async throws -> HestiaTurn {
        try await NativeOperation.run { [self] op in
            let data = try device.withCString { deviceID in
                try checked { hestia_confirm(handle, turn, deviceID, op, $0) }
            }
            return try JSONDecoder().decode(HestiaTurn.self, from: data)
        }
    }
    func addDocument(title: String, text: String) async throws {
        guard !title.contains("\0"), !text.contains("\0") else { throw GopherLLMError("Das Dokument muss Klartext sein.") }
        _ = try await NativeOperation.run { [self] op in
            try checked { hestia_add_document(handle, title, text, op, $0) }
        }
    }
}

private func take(_ pointer: UnsafeMutablePointer<CChar>?) -> String? {
    guard let pointer else { return nil }
    defer { gopherllm_free_string(pointer) }
    return String(cString: pointer)
}
private func checked(_ call: (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> UnsafeMutablePointer<CChar>?) throws -> Data {
    var error: UnsafeMutablePointer<CChar>?
    let output = take(call(&error))
    if let message = take(error) { throw GopherLLMError(message) }
    guard let output else { throw GopherLLMError("Hestia hat kein Ergebnis geliefert.") }
    return Data(output.utf8)
}

enum Credentials {
    private static func query(_ account: String) -> [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: Bundle.main.bundleIdentifier ?? "Hestia",
         kSecAttrAccount as String: account]
    }
    static func read(_ account: String) -> String {
        var request = query(account)
        request[kSecReturnData as String] = true
        var data: CFTypeRef?
        guard SecItemCopyMatching(request as CFDictionary, &data) == errSecSuccess,
              let value = data as? Data else { return "" }
        return String(decoding: value, as: UTF8.self)
    }
    static func save(_ value: String, account: String) throws {
        let request = query(account)
        if value.isEmpty {
            let result = SecItemDelete(request as CFDictionary)
            guard result == errSecSuccess || result == errSecItemNotFound else { throw GopherLLMError("Schlüsselbundfehler: \(result)") }
            return
        }
        let data = Data(value.utf8)
        var result = SecItemUpdate(request as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if result == errSecItemNotFound {
            var newItem = request
            newItem[kSecValueData as String] = data
            result = SecItemAdd(newItem as CFDictionary, nil)
        }
        guard result == errSecSuccess else { throw GopherLLMError("Schlüsselbundfehler: \(result)") }
    }
}
