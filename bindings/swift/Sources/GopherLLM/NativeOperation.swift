import Foundation
import GopherLLMCore

@_spi(AssistantHost) public final class NativeOperation: Sendable {
    private static let queue = DispatchQueue(label: "GopherLLM.NativeOperation", qos: .userInitiated, attributes: .concurrent)
    private let handle = gopherllm_operation_new()

    deinit { gopherllm_operation_free(handle) }

    public static func run<T: Sendable>(_ work: @escaping @Sendable (gopherllm_operation) throws -> T) async throws -> T {
        let operation = NativeOperation()
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            let result: Result<T, Error> = await withCheckedContinuation { continuation in
                queue.async {
                    continuation.resume(returning: Result { try work(operation.handle) })
                }
            }
            try Task.checkCancellation()
            return try result.get()
        } onCancel: {
            gopherllm_operation_cancel(operation.handle)
        }
    }
}

func withOptionalCString<T>(_ value: String?, _ body: (UnsafePointer<CChar>?) throws -> T) rethrows -> T {
    guard let value else { return try body(nil) }
    return try value.withCString { try body($0) }
}
