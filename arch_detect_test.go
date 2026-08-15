package gopherllm

import (
	"bytes"
	"strings"
	"testing"
)

// metaOnlyGGUF builds a tensor-less GGUF carrying just the given metadata —
// enough for ResolveArchitecture/ConfigFromGGUF, which never touch tensor
// data.
func metaOnlyGGUF(t *testing.T, kvs []ggufKV) *GGUFFile {
	t.Helper()
	g, err := ParseGGUFQuiet(buildGGUF(3, kvs, nil))
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestResolveArchitecture(t *testing.T) {
	u32 := func(key string, v uint32) ggufKV { return ggufKV{key, ggufU32, v} }
	str := func(key, v string) ggufKV { return ggufKV{key, ggufStr, v} }
	ns := func(prefix string) []ggufKV {
		return []ggufKV{u32(prefix+".block_count", 2), u32(prefix+".embedding_length", 8)}
	}
	cases := []struct {
		name       string
		kvs        []ggufKV
		wantArch   string
		wantPrefix string
	}{
		{"declared with own namespace", append(ns("qwen3"), str("general.architecture", "qwen3")), "qwen3", "qwen3"},
		{"llama3 label with llama namespace", append(ns("llama"), str("general.architecture", "llama3")), "llama3", "llama"},
		{"kimi_k2 label with deepseek2 namespace", append(ns("deepseek2"), str("general.architecture", "kimi_k2")), "kimi_k2", "deepseek2"},
		{"missing label detected from namespace", ns("qwen2"), "qwen2", "qwen2"},
		{"missing label without namespace defaults to llama", []ggufKV{str("general.name", "mystery")}, "llama", "llama"},
		{"unknown label adopts supported namespace", append(ns("llama"), str("general.architecture", "devstral")), "llama", "llama"},
		{"HF class name normalizes", append(ns("qwen2"), str("general.architecture", "Qwen2ForCausalLM")), "qwen2", "qwen2"},
		{"underscore variant keeps its namespace", append(ns("command_r"), str("general.architecture", "command_r")), "command-r", "command_r"},
		{"GPT2LMHeadModel strips HF suffix", append(ns("gpt2"), str("general.architecture", "GPT2LMHeadModel")), "gpt2", "gpt2"},
		{"deepseek-v3 aliases to deepseek2", append(ns("deepseek2"), str("general.architecture", "deepseek-v3")), "deepseek2", "deepseek2"},
		{"unsupported label with own namespace is preserved", append(ns("mamba"), str("general.architecture", "mamba")), "mamba", "mamba"},
		{"clip projector stays clip", []ggufKV{str("general.architecture", "clip"), u32("clip.vision.block_count", 24), str("clip.projector_type", "pixtral")}, "clip", "clip"},
		{"tokenizer and split keys never become namespaces", []ggufKV{str("general.architecture", ""), u32("split.count", 2), str("tokenizer.ggml.model", "llama")}, "llama", "llama"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			arch, prefix := ResolveArchitecture(metaOnlyGGUF(t, tc.kvs))
			if arch != tc.wantArch || prefix != tc.wantPrefix {
				t.Fatalf("ResolveArchitecture = (%q, %q), want (%q, %q)", arch, prefix, tc.wantArch, tc.wantPrefix)
			}
		})
	}
}

func TestCanonicalArchAlias(t *testing.T) {
	cases := map[string]string{
		"Qwen2ForCausalLM":      "qwen2",
		"Qwen2MoeForCausalLM":   "qwen2moe",
		"MistralForCausalLM":    "mistral",
		"PhiForCausalLM":        "phi2",
		"CohereForCausalLM":     "command-r",
		"GPTBigCodeForCausalLM": "starcoder",
		"GPTNeoXForCausalLM":    "gptneox",
		"BertModel":             "bert",
		"NomicBertModel":        "nomic-bert",
		"gpt_oss":               "gpt-oss",
		"granite-moe":           "granitemoe",
		"deepseek_v2":           "deepseek2",
		"qwen2.5":               "qwen2",
		"kimi-k2":               "kimi_k2",
		"QWEN3":                 "qwen3",
		"mamba":                 "", // Mamba-1 is deliberately unsupported; must not alias to mamba2
		"llava":                 "",
		"olmoe":                 "",
		"":                      "",
	}
	for label, want := range cases {
		if got := canonicalArchAlias(label); got != want {
			t.Errorf("canonicalArchAlias(%q) = %q, want %q", label, got, want)
		}
	}
}

func TestMissingArchitectureDetectedFromNamespaceLoads(t *testing.T) {
	data := buildTinyQwen3GGUF()
	g, err := ParseGGUFQuiet(data)
	if err != nil {
		t.Fatal(err)
	}
	delete(g.Metadata, "general.architecture")
	var log bytes.Buffer
	r, err := runnerFromParsedGGUF(data, g, false, LoadOptions{LogWriter: &log})
	if err != nil {
		t.Fatal(err)
	}
	if r.Architecture() != "qwen3" || r.config.Arch != "qwen3" {
		t.Fatalf("detected arch = %q / config %q, want qwen3", r.Architecture(), r.config.Arch)
	}
	if layer := r.standard.Layers[0]; layer.AttnQNorm == nil || layer.AttnKNorm == nil {
		t.Fatal("detected qwen3 must still load its QK-norm tensors")
	}
	if !strings.Contains(log.String(), "detected \"qwen3\"") {
		t.Fatalf("load log missing detection notice: %q", log.String())
	}
}

func TestUnknownLabelWithLlamaNamespaceLoadsAsLlama(t *testing.T) {
	data := buildTinyLlamaGGUF()
	g, err := ParseGGUFQuiet(data)
	if err != nil {
		t.Fatal(err)
	}
	g.Metadata["general.architecture"] = MetaValue{Kind: "str", Value: "devstral"}
	var log bytes.Buffer
	r, err := runnerFromParsedGGUF(data, g, false, LoadOptions{LogWriter: &log})
	if err != nil {
		t.Fatal(err)
	}
	if r.Architecture() != "llama" || r.config.Arch != "llama" || r.config.Dim != 8 {
		t.Fatalf("adopted arch = %q / config = {arch:%q dim:%d}, want llama with llama.* hparams", r.Architecture(), r.config.Arch, r.config.Dim)
	}
	if !strings.Contains(log.String(), "loading as \"llama\"") {
		t.Fatalf("load log missing adoption notice: %q", log.String())
	}
}

func TestUnsupportedArchitectureErrorKeepsExactLabel(t *testing.T) {
	g := metaOnlyGGUF(t, []ggufKV{
		{"general.architecture", ggufStr, "olmoe"},
		{"olmoe.block_count", ggufU32, uint32(2)},
		{"olmoe.embedding_length", ggufU32, uint32(8)},
	})
	_, err := runnerFromParsedGGUF(nil, g, false, LoadOptions{})
	if err == nil || !strings.Contains(err.Error(), "unsupported architecture: olmoe") {
		t.Fatalf("error = %v, want unsupported architecture: olmoe", err)
	}
}

func TestVisionProjectorGGUFGetsTargetedError(t *testing.T) {
	g := metaOnlyGGUF(t, []ggufKV{
		{"general.architecture", ggufStr, "clip"},
		{"clip.projector_type", ggufStr, "pixtral"},
		{"clip.has_vision_encoder", ggufBool, true},
		{"clip.vision.block_count", ggufU32, uint32(24)},
	})
	_, err := runnerFromParsedGGUF(nil, g, false, LoadOptions{})
	if err == nil || !strings.Contains(err.Error(), "vision projector") || !strings.Contains(err.Error(), "--mmproj") {
		t.Fatalf("error = %v, want mmproj-specific diagnostic", err)
	}
}

func TestConfigFromGGUFCommandRVariantAppliesCanonicalBehavior(t *testing.T) {
	g := metaOnlyGGUF(t, []ggufKV{
		{"general.architecture", ggufStr, "command_r"},
		{"command_r.block_count", ggufU32, uint32(2)},
		{"command_r.embedding_length", ggufU32, uint32(8)},
		{"command_r.attention.head_count", ggufU32, uint32(2)},
		{"command_r.logit_scale", ggufF32, float32(0.25)},
	})
	cfg := ConfigFromGGUF(g)
	if cfg.Arch != "command-r" {
		t.Fatalf("arch = %q, want command-r", cfg.Arch)
	}
	// Behavior must key on the canonical label even though the hparams live
	// under the variant namespace: Command-R inverts logit_scale and uses
	// LayerNorm with a forced parallel residual.
	if cfg.LogitScale != 4 {
		t.Fatalf("logit scale = %v, want reciprocal 4", cfg.LogitScale)
	}
	if !cfg.UseLayerNorm || !cfg.ParallelResidual {
		t.Fatalf("cfg = {layerNorm:%v parallel:%v}, want command-r behavior", cfg.UseLayerNorm, cfg.ParallelResidual)
	}
}

func TestAnalyzeReportsResolvedArchitecture(t *testing.T) {
	g, err := ParseGGUFQuiet(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	g.Metadata["general.architecture"] = MetaValue{Kind: "str", Value: "devstral"}
	a := AnalyzeGGUF(g, nil)
	if a.Architecture != "devstral" || a.LoadsAs != "llama" || !a.Supported {
		t.Fatalf("analysis = {arch:%q loadsAs:%q supported:%v}", a.Architecture, a.LoadsAs, a.Supported)
	}
	var buf bytes.Buffer
	a.WriteText(&buf)
	if !strings.Contains(buf.String(), "devstral (loads as llama, supported: true)") {
		t.Fatalf("report missing resolution: %q", buf.String())
	}
}
