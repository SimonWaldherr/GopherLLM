package gopherllm

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"os"
	"testing"
)

// TestVisionPlausibilityColorQuestion is an opt-in end-to-end plausibility
// smoke test: it loads a real text-decoder GGUF and a real paired Pixtral
// mmproj GGUF, asks the model to name the color of a synthetic solid-color
// PNG, and checks the expected color word appears in the reply.
//
// This is deliberately NOT a numerical-correctness proof — there is no
// automatable way to verify byte-exact parity against another Pixtral
// implementation (e.g. llama.cpp's) without depending on its code, which is
// out of scope for this project's no-third-party-code rule (see the project
// plan's M1.12). A passing run here means the whole pipeline (mmproj
// parsing, vision tower forward pass, spatial merge, projector, image
// placeholder token splicing, embedding substitution) produces coherent,
// correct-looking semantic output end to end — a real signal, just not a
// mathematical proof.
//
//	GOPHERLLM_VISION_MODEL=Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
//	GOPHERLLM_VISION_MMPROJ=mmproj-F16.gguf \
//	go test -run TestVisionPlausibilityColorQuestion -v .
func TestVisionPlausibilityColorQuestion(t *testing.T) {
	textPath := os.Getenv("GOPHERLLM_VISION_MODEL")
	visionPath := os.Getenv("GOPHERLLM_VISION_MMPROJ")
	if textPath == "" || visionPath == "" {
		t.Skip("set GOPHERLLM_VISION_MODEL and GOPHERLLM_VISION_MMPROJ to run this end-to-end plausibility test")
	}

	textData, err := os.ReadFile(textPath)
	if err != nil {
		t.Fatal(err)
	}
	visionData, err := os.ReadFile(visionPath)
	if err != nil {
		t.Fatal(err)
	}

	r, err := RunnerFromGGUFBytesWithVision(textData, visionData, LoadOptions{})
	if err != nil {
		t.Fatalf("RunnerFromGGUFBytesWithVision: %v", err)
	}
	defer r.Close()
	if !r.HasVision() {
		t.Fatal("HasVision() = false after loading a vision projector")
	}

	tests := []struct {
		name       string
		r, g, b    uint8
		wantSubstr string
	}{
		{"red", 220, 20, 20, "red"},
		{"blue", 20, 20, 220, "blue"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			img := image.NewNRGBA(image.Rect(0, 0, 64, 64))
			c := color.NRGBA{R: tc.r, G: tc.g, B: tc.b, A: 255}
			for y := 0; y < 64; y++ {
				for x := 0; x < 64; x++ {
					img.SetNRGBA(x, y, c)
				}
			}
			var buf bytes.Buffer
			if err := png.Encode(&buf, img); err != nil {
				t.Fatal(err)
			}

			msgs := []ChatMessage{
				UserMessageWithImages("What color is this image? Answer with one word.", ImageContent{Bytes: buf.Bytes()}),
			}
			res, err := r.GenerateChat(msgs, GenerationOptions{
				MaxTokens: 16,
				Sampler:   SamplerConfig{Temperature: 0, TopP: 1, RepeatPenalty: 1.1},
			})
			if err != nil {
				t.Fatalf("GenerateChat: %v", err)
			}
			t.Logf("model said: %q", res.Text)
			if !containsFold(res.Text, tc.wantSubstr) {
				t.Errorf("reply %q does not mention %q", res.Text, tc.wantSubstr)
			}
		})
	}
}

func containsFold(s, substr string) bool {
	return bytes.Contains(bytes.ToLower([]byte(s)), bytes.ToLower([]byte(substr)))
}
