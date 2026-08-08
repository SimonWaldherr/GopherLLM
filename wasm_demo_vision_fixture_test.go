package gopherllm

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

// buildTinyPixtralVisionGGUF builds a minimal but real Pixtral-projector
// GGUF (F32 throughout -- loadWeight's F32 branch needs no quantization,
// so a synthetic fixture doesn't need one either) whose ProjectionDim and
// img_break width (256) match buildTinyQuantizedMistralGGUF's Dim, so the
// two can be loaded together via RunnerFromGGUFBytesWithVision.
func buildTinyPixtralVisionGGUF() []byte {
	const (
		embLen    = 64
		heads     = 4
		headDim   = embLen / heads // 16
		ffnLen    = 64
		blocks    = 1
		patchSize = 4
		mergeSize = 2
		imageSize = 64
		projDim   = 256 // must match buildTinyQuantizedMistralGGUF's dim
	)
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "clip"},
		{"general.name", ggufStr, "tiny-vision"},
		{"clip.has_vision_encoder", ggufBool, true},
		{"clip.projector_type", ggufStr, "pixtral"},
		{"clip.use_silu", ggufBool, true},
		{"clip.vision.embedding_length", ggufU32, uint32(embLen)},
		{"clip.vision.feed_forward_length", ggufU32, uint32(ffnLen)},
		{"clip.vision.block_count", ggufU32, uint32(blocks)},
		{"clip.vision.attention.head_count", ggufU32, uint32(heads)},
		{"clip.vision.attention.head_dim", ggufU32, uint32(headDim)},
		{"clip.vision.attention.layer_norm_epsilon", ggufF32, float32(1e-5)},
		{"clip.vision.image_size", ggufU32, uint32(imageSize)},
		{"clip.vision.patch_size", ggufU32, uint32(patchSize)},
		{"clip.vision.spatial_merge_size", ggufU32, uint32(mergeSize)},
		{"clip.vision.projection_dim", ggufU32, uint32(projDim)},
	}

	f32t := func(name string, rows, cols, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(rows*cols, seed))}
	}
	f32tDims := func(name string, dims []uint64, seed int) ggufTensor {
		n := 1
		for _, d := range dims {
			n *= int(d)
		}
		return ggufTensor{name: name, dims: dims, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(n, seed))}
	}
	vec := func(name string, n, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(onesF32(n))}
	}

	mergedLen := embLen * mergeSize * mergeSize // 256

	tensors := []ggufTensor{
		f32tDims("v.patch_embd.weight", []uint64{uint64(patchSize), uint64(patchSize), 3, uint64(embLen)}, 1),
		vec("v.pre_ln.weight", embLen, 2),
		f32t("v.blk.0.attn_q.weight", embLen, embLen, 3),
		f32t("v.blk.0.attn_k.weight", embLen, embLen, 4),
		f32t("v.blk.0.attn_v.weight", embLen, embLen, 5),
		f32t("v.blk.0.attn_out.weight", embLen, embLen, 6),
		vec("v.blk.0.ln1.weight", embLen, 7),
		f32t("v.blk.0.ffn_gate.weight", ffnLen, embLen, 8),
		f32t("v.blk.0.ffn_up.weight", ffnLen, embLen, 9),
		f32t("v.blk.0.ffn_down.weight", embLen, ffnLen, 10),
		vec("v.blk.0.ln2.weight", embLen, 11),
		vec("mm.input_norm.weight", embLen, 12),
		f32t("mm.patch_merger.weight", embLen, mergedLen, 13),
		f32t("mm.1.weight", projDim, embLen, 14),
		f32t("mm.2.weight", projDim, projDim, 15),
		vec("v.token_embd.img_break", projDim, 16),
	}
	return buildGGUF(3, kvs, tensors)
}

// TestGenerateWasmDemoVisionFixture writes the tiny quantized text model
// (shared with TestGenerateWasmGPUFixture) plus a matching tiny Pixtral
// vision GGUF, for exercising the demo's vision bridge functions
// (gopherllm_loadModelWithVision, image content in gopherllm_generate) end
// to end without needing a multi-GB real model pair. Opt-in; regenerate with:
//
//	GOPHERLLM_GEN_WASM_VISION_FIXTURE=1 go test -run TestGenerateWasmDemoVisionFixture -v .
func TestGenerateWasmDemoVisionFixture(t *testing.T) {
	if os.Getenv("GOPHERLLM_GEN_WASM_VISION_FIXTURE") != "1" {
		t.Skip("set GOPHERLLM_GEN_WASM_VISION_FIXTURE=1 to (re)generate the tiny vision fixture")
	}
	// [INST]/[/INST] make chatTemplateKind() detect "mistral-inst" (via
	// Tokenizer.SpecialID, see runtime.go), the only template that accepts
	// ChatMessage.Images; [IMG]/[IMG_BREAK]/[IMG_END] are what
	// renderMistralInstMessages splices in around the image's patch
	// embeddings.
	textData := buildTinyQuantizedMistralGGUFWithSpecials([]string{"[INST]", "[/INST]", "[IMG]", "[IMG_BREAK]", "[IMG_END]"})
	visionData := buildTinyPixtralVisionGGUF()

	dir := filepath.Join("cmd", "gopherllm-wasm", "testdata", "harness")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Deliberately a different filename from TestGenerateWasmGPUFixture's
	// tiny-quant-model.gguf: that one has a 32-token vocab with no
	// [INST]-family specials (webgpu-forward-pass-test.html's hardcoded
	// GOLDEN_TEXT is pinned to it); this one has 5 extra special tokens
	// appended, changing the vocabulary size and thus, in principle, the
	// arg-max output at every decode step. Sharing a filename between the
	// two would make whichever test ran last silently invalidate the
	// other's golden-output harness.
	if err := os.WriteFile(filepath.Join(dir, "tiny-quant-model-vision.gguf"), textData, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tiny-vision.gguf"), visionData, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote tiny-quant-model-vision.gguf (%d bytes) and tiny-vision.gguf (%d bytes)", len(textData), len(visionData))

	r, err := RunnerFromGGUFBytesWithVision(textData, visionData, LoadOptions{LogWriter: os.Stderr})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if !r.HasVision() {
		t.Fatal("HasVision() = false after loading a vision projector")
	}

	img := image.NewNRGBA(image.Rect(0, 0, 64, 64))
	red := color.NRGBA{R: 220, G: 20, B: 20, A: 255}
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			img.SetNRGBA(x, y, red)
		}
	}
	var pngBuf bytes.Buffer
	if err := png.Encode(&pngBuf, img); err != nil {
		t.Fatal(err)
	}
	res, err := r.GenerateChat([]ChatMessage{
		UserMessageWithImages("Hello", ImageContent{Bytes: pngBuf.Bytes()}),
	}, GenerationOptions{
		MaxTokens: 8,
		Sampler:   SamplerConfig{Temperature: 0, TopP: 1, RepeatPenalty: 1.1},
	})
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("VISION_GOLDEN_TEXT=%q\n", res.Text)
}
