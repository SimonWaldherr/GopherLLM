import AVFoundation
import Foundation

/// The audio tap owns the converter. Only the accumulator crosses threads.
final class Microphone {
    enum Mode { case manual, wake }
    private let engine = AVAudioEngine()
    private let accumulator: ClipAccumulator

    init(mode: Mode, onClip: @escaping @Sendable ([Float]) -> Void) throws {
        accumulator = ClipAccumulator(mode: mode, onClip: onClip)
        let node = engine.inputNode
        let input = node.outputFormat(forBus: 0)
        guard input.channelCount > 0, input.sampleRate > 0,
              let mono = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: input, to: mono) else {
            throw CaptureError.badFormat
        }
        let accumulator = accumulator
        node.installTap(onBus: 0, bufferSize: 2048, format: input) { buffer, _ in
            let frames = AVAudioFrameCount(ceil(Double(buffer.frameLength) * 16_000 / input.sampleRate) + 32)
            guard let converted = AVAudioPCMBuffer(pcmFormat: mono, frameCapacity: frames) else { return }
            var sent = false
            var error: NSError?
            converter.convert(to: converted, error: &error) { _, status in
                if sent { status.pointee = .noDataNow; return nil }
                sent = true
                status.pointee = .haveData
                return buffer
            }
            guard error == nil, let samples = converted.floatChannelData?.pointee else { return }
            accumulator.push(UnsafeBufferPointer(start: samples, count: Int(converted.frameLength)))
        }
        do { try engine.start() } catch {
            node.removeTap(onBus: 0)
            throw error
        }
    }

    func stop() -> [Float] {
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
        return accumulator.finish()
    }

    private enum CaptureError: LocalizedError {
        case badFormat
        var errorDescription: String? { "Das Mikrofon liefert kein unterstütztes Audioformat." }
    }
}

private final class ClipAccumulator: @unchecked Sendable {
    private let lock = NSLock()
    private let mode: Microphone.Mode
    private let onClip: @Sendable ([Float]) -> Void
    private var samples: [Float] = []
    private var preRoll: [Float] = []
    private var quiet = 0
    private var overflow = false
    private var stopped = false

    init(mode: Microphone.Mode, onClip: @escaping @Sendable ([Float]) -> Void) {
        self.mode = mode
        self.onClip = onClip
    }

    func push(_ chunk: UnsafeBufferPointer<Float>) {
        guard !chunk.isEmpty else { return }
        let clip: [Float]? = lock.withLock {
            guard !stopped else { return nil }
            if mode == .manual {
                samples.append(contentsOf: chunk.prefix(480_000 - samples.count).map { min(1, max(-1, $0)) })
                if samples.count >= 480_000 {
                    stopped = true
                    return samples
                }
                return nil
            }

            let loud = chunk.reduce(Float.zero) { $0 + $1 * $1 } / Float(chunk.count) > 0.00016
            if samples.isEmpty && !overflow {
                if !loud {
                    preRoll.append(contentsOf: chunk)
                    if preRoll.count > 2400 { preRoll.removeFirst(preRoll.count - 2400) }
                    return nil
                }
                samples = preRoll
                preRoll = []
            }
            quiet = loud ? 0 : quiet + chunk.count
            if !overflow {
                samples.append(contentsOf: chunk)
                if samples.count > 128_000 { overflow = true; samples = [] }
            }
            guard quiet >= 8800 else { return nil }
            let result = overflow || samples.count - quiet < 2400 ? nil : samples.map { min(1, max(-1, $0)) }
            samples = []
            quiet = 0
            overflow = false
            return result
        }
        if let clip { onClip(clip) }
    }

    func finish() -> [Float] {
        lock.withLock {
            stopped = true
            defer { samples = []; preRoll = [] }
            return mode == .manual ? samples : []
        }
    }
}
