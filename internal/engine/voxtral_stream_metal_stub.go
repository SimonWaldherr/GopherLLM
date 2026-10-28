//go:build !darwin || !cgo || !metal

package engine

func newVoxtralFastDecoder(cfg VoxtralRealtimeConfig, w VoxtralRealtimeWeights, state *VoxtralRealtimeDecoderState) voxtralFastDecoder {
	return nil
}
