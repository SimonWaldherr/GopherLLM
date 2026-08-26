package gopherllm

import "testing"

func testAcceleratedWeight() Weight {
	return Weight{Metal: &MetalWeight{}, GPU: &GPUWeight{}}
}

func assertAcceleratorReleased(t *testing.T, name string, w *Weight) {
	t.Helper()
	if w.Metal != nil || w.GPU != nil {
		t.Fatalf("%s retains accelerator resources: Metal=%p GPU=%p", name, w.Metal, w.GPU)
	}
}

func TestReleaseModelWeightResourcesIncludesPositionEmbedding(t *testing.T) {
	weights := ModelWeights{
		TokenEmbd:    testAcceleratedWeight(),
		PositionEmbd: testAcceleratedWeight(),
		Output:       testAcceleratedWeight(),
		Layers: []LayerWeights{{
			WQ:      testAcceleratedWeight(),
			WGateUp: testAcceleratedWeight(),
			MLA:     &MLAAttentionWeights{Q: testAcceleratedWeight(), KB: ExpertWeight{Weight: testAcceleratedWeight()}},
			MoE:     &SparseMoEWeights{Router: testAcceleratedWeight(), Gate: ExpertWeight{Weight: testAcceleratedWeight()}},
		}},
	}

	releaseModelMetalWeights(&weights)
	assertAcceleratorReleased(t, "token embedding", &weights.TokenEmbd)
	assertAcceleratorReleased(t, "position embedding", &weights.PositionEmbd)
	assertAcceleratorReleased(t, "output", &weights.Output)
	assertAcceleratorReleased(t, "layer Q", &weights.Layers[0].WQ)
	assertAcceleratorReleased(t, "layer gate/up", &weights.Layers[0].WGateUp)
	assertAcceleratorReleased(t, "MLA Q", &weights.Layers[0].MLA.Q)
	assertAcceleratorReleased(t, "MLA KB", &weights.Layers[0].MLA.KB.Weight)
	assertAcceleratorReleased(t, "MoE router", &weights.Layers[0].MoE.Router)
	assertAcceleratorReleased(t, "MoE gate", &weights.Layers[0].MoE.Gate.Weight)
}

func TestReleaseSpecializedWeightResourcesIncludesGPU(t *testing.T) {
	t.Run("BERT", func(t *testing.T) {
		weights := BERTWeights{
			PositionEmbd: testAcceleratedWeight(),
			Layers:       []BERTLayerWeights{{QKV: testAcceleratedWeight(), FFNGate: testAcceleratedWeight()}},
		}
		releaseBERTMetalWeights(&weights)
		assertAcceleratorReleased(t, "position embedding", &weights.PositionEmbd)
		assertAcceleratorReleased(t, "QKV", &weights.Layers[0].QKV)
		assertAcceleratorReleased(t, "FFN gate", &weights.Layers[0].FFNGate)
	})

	t.Run("Gemma4", func(t *testing.T) {
		weights := Gemma4Weights{
			Native:   true,
			PerLayer: &Gemma4PerLayerWeights{ModelProj: testAcceleratedWeight()},
			Layers:   []Gemma4LayerWeights{{AttnQ: testAcceleratedWeight(), PerLayerProj: testAcceleratedWeight()}},
			MoE:      []*Gemma4MoEWeights{{Router: testAcceleratedWeight(), Gate: ExpertWeight{Weight: testAcceleratedWeight()}}},
		}
		releaseGemma4MetalWeights(&weights)
		assertAcceleratorReleased(t, "per-layer projection", &weights.PerLayer.ModelProj)
		assertAcceleratorReleased(t, "attention Q", &weights.Layers[0].AttnQ)
		assertAcceleratorReleased(t, "layer projection", &weights.Layers[0].PerLayerProj)
		assertAcceleratorReleased(t, "MoE router", &weights.MoE[0].Router)
		assertAcceleratorReleased(t, "MoE gate", &weights.MoE[0].Gate.Weight)
	})

	t.Run("Mamba2", func(t *testing.T) {
		weights := Mamba2Weights{Layers: []Mamba2LayerWeights{{Mamba: NemotronMambaWeights{In: testAcceleratedWeight(), Conv: testAcceleratedWeight(), Out: testAcceleratedWeight()}}}}
		releaseMamba2MetalWeights(&weights)
		assertAcceleratorReleased(t, "Mamba input", &weights.Layers[0].Mamba.In)
		assertAcceleratorReleased(t, "Mamba convolution", &weights.Layers[0].Mamba.Conv)
		assertAcceleratorReleased(t, "Mamba output", &weights.Layers[0].Mamba.Out)
	})

	t.Run("NemotronH", func(t *testing.T) {
		latent := testAcceleratedWeight()
		weights := NemotronHWeights{Layers: []NemotronHLayerWeights{{
			Attention: NemotronAttentionWeights{Q: testAcceleratedWeight()},
			Mamba:     NemotronMambaWeights{In: testAcceleratedWeight()},
			MoE: NemotronMoEWeights{
				Router:   testAcceleratedWeight(),
				Up:       ExpertWeight{Weight: testAcceleratedWeight()},
				LatentIn: &latent,
				SharedUp: &latent,
			},
			DenseFFN: NemotronDenseFFNWeights{Down: testAcceleratedWeight()},
		}}}
		releaseNemotronHMetalWeights(&weights)
		layer := &weights.Layers[0]
		assertAcceleratorReleased(t, "attention Q", &layer.Attention.Q)
		assertAcceleratorReleased(t, "Mamba input", &layer.Mamba.In)
		assertAcceleratorReleased(t, "MoE router", &layer.MoE.Router)
		assertAcceleratorReleased(t, "MoE up", &layer.MoE.Up.Weight)
		assertAcceleratorReleased(t, "MoE latent/shared", layer.MoE.LatentIn)
		assertAcceleratorReleased(t, "dense FFN", &layer.DenseFFN.Down)
	})

	t.Run("Qwen35", func(t *testing.T) {
		shared := testAcceleratedWeight()
		weights := Qwen35Weights{
			MTP: &Qwen35MTPWeights{EHProj: testAcceleratedWeight(), FFN: Qwen35FFNWeights{Gate: testAcceleratedWeight()}},
			Layers: []Qwen35LayerWeights{{
				Attention: Qwen35AttentionWeights{Q: testAcceleratedWeight()},
				DeltaNet:  Qwen35DeltaNetWeights{QKVConv: testAcceleratedWeight(), Out: testAcceleratedWeight()},
				FFN: Qwen35FFNWeights{
					Gate: testAcceleratedWeight(),
					MoE:  &SparseMoEWeights{Router: testAcceleratedWeight(), SharedUp: &shared},
				},
			}},
		}
		releaseQwen35MetalWeights(&weights)
		assertAcceleratorReleased(t, "MTP projection", &weights.MTP.EHProj)
		assertAcceleratorReleased(t, "MTP FFN", &weights.MTP.FFN.Gate)
		layer := &weights.Layers[0]
		assertAcceleratorReleased(t, "attention Q", &layer.Attention.Q)
		assertAcceleratorReleased(t, "DeltaNet QKV", &layer.DeltaNet.QKVConv)
		assertAcceleratorReleased(t, "DeltaNet output", &layer.DeltaNet.Out)
		assertAcceleratorReleased(t, "MoE router", &layer.FFN.MoE.Router)
		assertAcceleratorReleased(t, "MoE shared", layer.FFN.MoE.SharedUp)
	})
}

func TestRunnerCloseReleasesVisionWeightResources(t *testing.T) {
	vision := &PixtralVisionWeights{
		PatchEmbd:   testAcceleratedWeight(),
		Layers:      []PixtralVisionLayerWeights{{Q: testAcceleratedWeight(), FFNDown: testAcceleratedWeight()}},
		PatchMerger: testAcceleratedWeight(),
		Proj1:       testAcceleratedWeight(),
		Proj2:       testAcceleratedWeight(),
	}
	r := &Runner{vision: vision, visionConfig: PixtralVisionConfig{EmbeddingLength: 1}}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if r.vision != nil || r.HasVision() {
		t.Fatal("Close retained the vision tower")
	}
	if r.visionConfig != (PixtralVisionConfig{}) {
		t.Fatalf("Close retained vision config: %+v", r.visionConfig)
	}
	assertAcceleratorReleased(t, "patch embedding", &vision.PatchEmbd)
	assertAcceleratorReleased(t, "attention Q", &vision.Layers[0].Q)
	assertAcceleratorReleased(t, "FFN down", &vision.Layers[0].FFNDown)
	assertAcceleratorReleased(t, "patch merger", &vision.PatchMerger)
	assertAcceleratorReleased(t, "projector 1", &vision.Proj1)
	assertAcceleratorReleased(t, "projector 2", &vision.Proj2)
}

func TestRunnerCloseDropsWeightContainers(t *testing.T) {
	weight := func() Weight { return Weight{F32: []float32{1}, Raw: []byte{1}, Prepared: &PreparedQuantizedWeight{}} }
	r := &Runner{
		standard:  ModelWeights{TokenEmbd: weight()},
		gptOss:    GptOssWeights{Standard: ModelWeights{TokenEmbd: weight()}},
		gemma4:    Gemma4Weights{TokenEmbd: weight()},
		nemotronH: NemotronHWeights{TokenEmbd: weight()},
		mamba2:    Mamba2Weights{TokenEmbd: weight()},
		bert:      BERTWeights{TokenEmbd: weight()},
		qwen35:    Qwen35Weights{TokenEmbd: weight()},
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	for name, w := range map[string]Weight{
		"standard":   r.standard.TokenEmbd,
		"gpt-oss":    r.gptOss.Standard.TokenEmbd,
		"gemma4":     r.gemma4.TokenEmbd,
		"nemotron-h": r.nemotronH.TokenEmbd,
		"mamba2":     r.mamba2.TokenEmbd,
		"bert":       r.bert.TokenEmbd,
		"qwen35":     r.qwen35.TokenEmbd,
	} {
		if w.F32 != nil || w.Raw != nil || w.Prepared != nil {
			t.Fatalf("Close retained %s tensor storage: %+v", name, w)
		}
	}
}

func TestRunnerFromGGUFBytesWithMetalCopiesBorrowedViews(t *testing.T) {
	textData := buildTinyQuantizedMistralGGUF()
	visionData := buildTinyPixtralVisionGGUF()
	gguf, err := ParseGGUFQuiet(textData)
	if err != nil {
		t.Fatal(err)
	}
	var token TensorInfo
	for _, info := range gguf.Tensors {
		if info.Name == "token_embd.weight" {
			token = info
			break
		}
	}
	if token.Name == "" {
		t.Fatal("token embedding tensor missing from fixture")
	}
	offset := gguf.DataOffset + int(token.Offset)
	if offset < 0 || offset >= len(textData) {
		t.Fatalf("token embedding offset %d outside %d-byte fixture", offset, len(textData))
	}

	r, err := RunnerFromGGUFBytesWithOptions(textData, LoadOptions{
		BorrowQuantized:      true,
		UseMetal:             true,
		VisionProjectorBytes: visionData,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	raw := r.standard.TokenEmbd.Raw
	if len(raw) == 0 {
		t.Fatal("token embedding was not loaded as a quantized raw weight")
	}
	if &raw[0] == &textData[offset] {
		t.Fatal("UseMetal retained a byte-backed GGUF tensor instead of making an owned copy")
	}
	before := raw[0]
	textData[offset] ^= 0xff
	if got := r.standard.TokenEmbd.Raw[0]; got != before {
		t.Fatalf("owned token weight changed after caller mutated source bytes: got %d want %d", got, before)
	}
	if r.vision == nil || r.vision.PatchEmbd.F32 == nil || r.vision.PatchEmbd.Raw != nil {
		t.Fatal("vision projector did not follow the effective non-borrowing load mode")
	}
}
