//go:build !darwin || !cgo || !metal

package gopherllm

func newVoxtralFastDecoder(cfg VoxtralRealtimeConfig, w VoxtralRealtimeWeights, state *VoxtralRealtimeDecoderState) voxtralFastDecoder {
	return nil
}
