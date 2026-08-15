package gopherllm

import (
	"sort"
	"strings"
)

// This file resolves which architecture a GGUF should be loaded as. The
// `general.architecture` string is authoritative when it is present, known,
// and consistent with the rest of the file — but real-world GGUFs regularly
// violate one of those three assumptions:
//
//   - the key is missing entirely (hand-assembled or minimal exporters);
//   - the label is a converter-specific spelling of a known architecture
//     (Hugging Face class names like "Qwen2ForCausalLM", underscore/hyphen
//     variants like "command_r", model-type labels like "kimi_k2");
//   - the label names a fine-tune while the hyperparameters were written
//     under the canonical llama.cpp namespace of the base architecture.
//
// ResolveArchitecture handles all three by combining the declared label with
// the hyperparameter namespace that is actually present in the metadata.

// archHParamMarkers are metadata key suffixes whose presence identifies
// "<prefix>" as a hyperparameter namespace. Every llama.cpp-style converter
// writes at least block_count/embedding_length/context_length under the
// architecture's own prefix, so requiring one of these makes the scan precise:
// keys like "tokenizer.ggml.model" or "split.count" can never match, and
// vision-projector keys ("clip.vision.block_count") are rejected because their
// prefix contains a dot.
var archHParamMarkers = [...]string{
	".block_count",
	".embedding_length",
	".context_length",
	".attention.head_count",
	".feed_forward_length",
	".vocab_size",
}

// reservedMetadataNamespaces can never be architecture namespaces even if a
// marker-shaped key appears under them.
var reservedMetadataNamespaces = map[string]bool{
	"general":   true,
	"tokenizer": true,
	"split":     true,
	"quantize":  true,
	"adapter":   true,
	"training":  true,
	"clip":      true, // vision projectors are loaded via LoadOptions.VisionProjector*, never as text models
}

// archLabelAliases maps architecture labels seen in the wild to the canonical
// llama.cpp label that GopherLLM's loaders and graph switches key on. Every
// entry must be graph-identical to its target — this table is for spelling
// variants and converter model-type labels, not for "close enough" families.
// Formatting-only variants (case, hyphen/underscore, Hugging Face class-name
// suffixes) are handled generically by canonicalArchAlias and do not need
// entries here.
var archLabelAliases = map[string]string{
	"phi":        "phi2",      // HF PhiForCausalLM covers Phi-1/1.5/2; all use llama.cpp's phi2 graph
	"cohere":     "command-r", // HF CohereForCausalLM
	"commandr":   "command-r",
	"gptbigcode": "starcoder", // HF GPTBigCodeForCausalLM
	"gptoss":     "gpt-oss",
	"nomicbert":  "nomic-bert",
	"kimik2":     "kimi_k2",
	"deepseekv2": "deepseek2", // DeepSeek-V2/V3 GGUFs canonically declare deepseek2
	"deepseekv3": "deepseek2",
	"qwen2.5":    "qwen2", // Qwen2.5 checkpoints are qwen2-architecture
	"qwen25":     "qwen2",
}

// canonicalArchAlias maps a non-canonical architecture label to the supported
// canonical label it is a spelling of, or "" when the label is genuinely
// unknown. It tries, in order: hyphen/underscore/separator-free variants of
// the lowercased label, then the same variants with a Hugging Face class-name
// suffix ("...ForCausalLM", "...LMHeadModel", ...) stripped. Each attempt is
// accepted
// only when it lands on ArchitectureSupported or an explicit alias, so the
// generic suffix stripping cannot invent a label the loader would reject.
func canonicalArchAlias(label string) string {
	check := func(s string) string {
		stripped := strings.ReplaceAll(strings.ReplaceAll(s, "-", ""), "_", "")
		for _, variant := range []string{s, strings.ReplaceAll(s, "_", "-"), strings.ReplaceAll(s, "-", "_"), stripped} {
			if ArchitectureSupported(variant) {
				return variant
			}
			if alias, ok := archLabelAliases[variant]; ok {
				return alias
			}
		}
		return ""
	}
	lower := strings.ToLower(strings.TrimSpace(label))
	if lower == "" {
		return ""
	}
	if hit := check(lower); hit != "" {
		return hit
	}
	for _, suffix := range []string{"forcausallm", "forconditionalgeneration", "forsequenceclassification", "lmheadmodel", "model"} {
		trimmed := strings.TrimSuffix(lower, suffix)
		if trimmed == lower || trimmed == "" {
			continue
		}
		if hit := check(trimmed); hit != "" {
			return hit
		}
	}
	return ""
}

// namespaceHasHParams reports whether at least one hyperparameter marker key
// exists under prefix.
func namespaceHasHParams(gguf *GGUFFile, prefix string) bool {
	for _, marker := range archHParamMarkers {
		if _, ok := gguf.Metadata[prefix+marker]; ok {
			return true
		}
	}
	return false
}

// hparamNamespaceCandidates scans the metadata for hyperparameter namespaces
// and returns them best-first: most marker keys, then supported architectures
// before unsupported ones, then lexicographic for determinism. Text-model
// GGUFs have exactly one candidate in practice; the ranking only matters for
// malformed files that carry stray keys from several namespaces.
func hparamNamespaceCandidates(gguf *GGUFFile) []string {
	hits := map[string]int{}
	for key := range gguf.Metadata {
		for _, marker := range archHParamMarkers {
			if !strings.HasSuffix(key, marker) {
				continue
			}
			prefix := strings.TrimSuffix(key, marker)
			if prefix == "" || strings.Contains(prefix, ".") || reservedMetadataNamespaces[prefix] {
				continue
			}
			hits[prefix]++
			break
		}
	}
	candidates := make([]string, 0, len(hits))
	for prefix := range hits {
		candidates = append(candidates, prefix)
	}
	sort.Slice(candidates, func(i, j int) bool {
		if hits[candidates[i]] != hits[candidates[j]] {
			return hits[candidates[i]] > hits[candidates[j]]
		}
		si, sj := ArchitectureSupported(candidates[i]), ArchitectureSupported(candidates[j])
		if si != sj {
			return si
		}
		return candidates[i] < candidates[j]
	})
	return candidates
}

// ResolveArchitecture determines the architecture a GGUF loads as and the
// metadata namespace its hyperparameters live under. The two differ whenever
// a converter kept a non-canonical public label: kimi_k2 files with
// deepseek2.* hparams resolve to ("kimi_k2", "deepseek2"), llama3 files with
// llama.* hparams to ("llama3", "llama"), and a "command_r" file with
// command_r.* hparams to ("command-r", "command_r").
//
// Resolution rules, in order:
//
//  1. The hyperparameter namespace is the declared label's own when its
//     marker keys exist, otherwise the best namespace actually present in
//     the metadata (this generalizes the historic llama2/llama3/kimi_k2
//     special cases to every converter that mislabels a file).
//  2. A missing/empty general.architecture takes the detected namespace as
//     its label, keeping the historic "llama" default only when the file has
//     no hyperparameter namespace at all.
//  3. A label that is not in ArchitectureSupported is normalized via
//     canonicalArchAlias (case, hyphen/underscore, HF class-name suffixes,
//     known alias table); if that fails but the namespace belongs to a
//     supported architecture, the file loads as the namespace architecture.
//
// Labels that are already supported are never rewritten, so alias labels
// with dedicated behavior (llama2/llama3, kimi_k2) keep their identity.
func ResolveArchitecture(gguf *GGUFFile) (arch, prefix string) {
	declared, _ := gguf.GetString("general.architecture")
	declared = strings.TrimSpace(declared)
	if declared != "" && namespaceHasHParams(gguf, declared) {
		prefix = declared
	} else if candidates := hparamNamespaceCandidates(gguf); len(candidates) > 0 {
		prefix = candidates[0]
	}
	arch = declared
	if arch == "" {
		if prefix != "" {
			arch = prefix
		} else {
			arch = "llama"
		}
	}
	if prefix == "" {
		prefix = arch
	}
	if !ArchitectureSupported(arch) {
		if alias := canonicalArchAlias(arch); alias != "" {
			arch = alias
		} else if arch != prefix && ArchitectureSupported(prefix) {
			arch = prefix
		}
	}
	return arch, prefix
}

// isVisionProjectorGGUF reports whether this GGUF is a CLIP-style vision
// projector ("mmproj") file rather than a text model. Projector files either
// declare the clip architecture or carry the clip.* projector keys.
func isVisionProjectorGGUF(gguf *GGUFFile, arch string) bool {
	if arch == "clip" {
		return true
	}
	if _, ok := gguf.Metadata["clip.projector_type"]; ok {
		return true
	}
	return gguf.GetBool("clip.has_vision_encoder", false)
}
