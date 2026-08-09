package gopherllm

import (
	"bytes"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

func testEntry(id, arch string, projector bool) ModelEntry {
	return ModelEntry{
		ID:           id,
		Repository:   "repo",
		FileName:     id + ".gguf",
		Path:         "/models/" + id + ".gguf",
		SizeBytes:    1024,
		Architecture: arch,
		IsProjector:  projector,
		IsSupported:  ArchitectureSupported(arch),
	}
}

func TestSelectModelIgnoresProjectorMatchesWhenTextModelExists(t *testing.T) {
	entries := []ModelEntry{
		testEntry("mistral/mmproj-mistral", "clip", true),
		testEntry("mistral/mistral-Q4_K_M", "mistral3", false),
	}
	selected, err := SelectModel(entries, "mistral")
	if err != nil {
		t.Fatal(err)
	}
	if selected.ID != "mistral/mistral-Q4_K_M" {
		t.Fatalf("selected %q", selected.ID)
	}
}

func TestSelectModelReportsAmbiguousTextMatches(t *testing.T) {
	entries := []ModelEntry{
		testEntry("llama-3/llama-3-Q4_K_M", "llama", false),
		testEntry("llama-2/llama-2-Q4_K_M", "llama", false),
	}
	if _, err := SelectModel(entries, "llama"); err == nil {
		t.Fatal("expected ambiguous selector error")
	}
}

func TestArchitectureSupportedCoversImplementedLoaders(t *testing.T) {
	for _, arch := range []string{"llama", "llama2", "llama3", "mistral", "mistral3", "mixtral", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "qwen35", "qwen35moe", "deepseek2", "kimi_k2", "phi2", "phi3", "granite", "exaone", "exaone4", "smollm3", "olmo2", "internlm2", "stablelm", "gpt-oss", "gemma", "gemma2", "gemma3", "gemma4", "bert", "nomic-bert"} {
		if !ArchitectureSupported(arch) {
			t.Fatalf("ArchitectureSupported(%q) = false, want true", arch)
		}
	}
	for _, arch := range []string{"olmoe"} {
		if ArchitectureSupported(arch) {
			t.Fatalf("ArchitectureSupported(%q) = true, want false", arch)
		}
	}
}

func TestModelSupportsReasoning(t *testing.T) {
	plain := &GGUFFile{Metadata: map[string]MetaValue{}}
	thinkingTemplate := &GGUFFile{Metadata: map[string]MetaValue{
		"tokenizer.chat_template": {Kind: "str", Value: "{% if true %}<think>{% endif %}"},
	}}
	for _, test := range []struct {
		name, arch, file, model string
		gguf                    *GGUFFile
		want                    bool
	}{
		{"qwen3 capability", "qwen3", "model.gguf", "Qwen3", plain, true},
		{"template capability", "mistral3", "model.gguf", "Mistral", thinkingTemplate, true},
		{"named reasoning model", "mistral3", "model.gguf", "Ministral Reasoning", plain, true},
		{"plain instruct model", "mistral3", "model.gguf", "Ministral Instruct", plain, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := modelSupportsReasoning(test.arch, test.file, test.model, test.gguf); got != test.want {
				t.Fatalf("modelSupportsReasoning() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestPromptModelSelectionAcceptsNumber(t *testing.T) {
	entries := []ModelEntry{
		testEntry("alpha/alpha-Q4_K_M", "llama", false),
		testEntry("beta/beta-Q4_K_M", "mistral3", false),
	}
	var out bytes.Buffer

	selected, err := PromptModelSelection("/models", entries, strings.NewReader("2\n"), &out)
	if err != nil {
		t.Fatal(err)
	}
	if selected.ID != "beta/beta-Q4_K_M" {
		t.Fatalf("selected %q", selected.ID)
	}
	if !strings.Contains(out.String(), "Found 2 supported GGUF models") {
		t.Fatalf("selection menu was not printed: %q", out.String())
	}
}

func TestPromptModelSelectionAcceptsUniqueTextFilter(t *testing.T) {
	entries := []ModelEntry{
		testEntry("alpha/alpha-Q4_K_M", "llama", false),
		testEntry("beta/beta-Q4_K_M", "mistral3", false),
	}
	var out bytes.Buffer

	selected, err := PromptModelSelection("/models", entries, strings.NewReader("beta\n"), &out)
	if err != nil {
		t.Fatal(err)
	}
	if selected.ID != "beta/beta-Q4_K_M" {
		t.Fatalf("selected %q", selected.ID)
	}
}

func TestPromptModelSelectionCanAbort(t *testing.T) {
	entries := []ModelEntry{
		testEntry("alpha/alpha-Q4_K_M", "llama", false),
		testEntry("beta/beta-Q4_K_M", "mistral3", false),
	}

	var out bytes.Buffer
	if _, err := PromptModelSelection("/models", entries, strings.NewReader("q\n"), &out); err == nil {
		t.Fatal("expected abort error")
	}
}

func TestDefaultModelDirPrefersEnvironment(t *testing.T) {
	t.Setenv("GOPHERLLM_MODEL_DIR", "  /models/custom  ")
	t.Setenv("RUSTY_LLM_MODEL_DIR", "")
	if got := DefaultModelDir(); got != "/models/custom" {
		t.Fatalf("DefaultModelDir() = %q", got)
	}
	// The deprecated pre-rename spelling still works as a fallback...
	t.Setenv("GOPHERLLM_MODEL_DIR", "")
	t.Setenv("RUSTY_LLM_MODEL_DIR", "/models/legacy")
	if got := DefaultModelDir(); got != "/models/legacy" {
		t.Fatalf("DefaultModelDir() deprecated fallback = %q", got)
	}
	// ...but the preferred GOPHERLLM_MODEL_DIR wins when both are set.
	t.Setenv("GOPHERLLM_MODEL_DIR", "/models/new")
	if got := DefaultModelDir(); got != "/models/new" {
		t.Fatalf("DefaultModelDir() with both set = %q", got)
	}
	t.Setenv("GOPHERLLM_MODEL_DIR", "")
	t.Setenv("RUSTY_LLM_MODEL_DIR", "")
	home := t.TempDir()
	t.Setenv("HOME", home)
	if got := DefaultModelDir(); got != filepath.Join(home, lmStudioCommunitySubdir) {
		t.Fatalf("DefaultModelDir() = %q", got)
	}
	if err := os.MkdirAll(filepath.Join(home, lmStudioModelsSubdir), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := DefaultModelDir(); got != filepath.Join(home, lmStudioModelsSubdir) {
		t.Fatalf("DefaultModelDir() library root = %q", got)
	}
}

func TestCatalogHelpers(t *testing.T) {
	if got := (ModelEntry{IsProjector: true, IsSupported: true}).Status(); got != "projector" {
		t.Fatalf("projector status = %q", got)
	}
	if got := (ModelEntry{IsSupported: true}).Status(); got != "supported" {
		t.Fatalf("supported status = %q", got)
	}
	if got := (ModelEntry{}).Status(); got != "unsupported" {
		t.Fatalf("unsupported status = %q", got)
	}
	if got := truncate("abcdef", 4); got != "abc~" {
		t.Fatalf("truncate = %q", got)
	}
	if got := truncate("abcdef", 1); got != "~" {
		t.Fatalf("truncate short = %q", got)
	}
	entries := []ModelEntry{{ID: "first", Path: "/models/first.gguf"}, {ID: "second", Path: "/models/second.gguf"}}
	if got := modelMenuIndex(entries, entries[1]); got != 2 {
		t.Fatalf("modelMenuIndex = %d", got)
	}
	if got := modelMenuIndex(entries, ModelEntry{ID: "missing"}); got != 0 {
		t.Fatalf("missing modelMenuIndex = %d", got)
	}
	if want, got := filepath.Dir(entries[0].Path), modelDirFromEntries(entries); got != want {
		t.Fatalf("modelDirFromEntries = %q, want %q", got, want)
	}
	if got := modelDirFromEntries(nil); got != "the model directory" {
		t.Fatalf("empty modelDirFromEntries = %q", got)
	}
}

func TestModelSortKeyPutsSupportedModelsFirstAndUsesDisplayName(t *testing.T) {
	entries := []ModelEntry{
		{ID: "repo/unsupported", ModelName: "Aardvark", IsSupported: false},
		{ID: "repo/zeta", ModelName: "zeta", IsSupported: true},
		{ID: "repo/alpha-b", ModelName: "Alpha", IsSupported: true},
		{ID: "repo/alpha-a", ModelName: "alpha", IsSupported: true},
	}
	sort.Slice(entries, func(i, j int) bool {
		return modelSortKey(entries[i]) < modelSortKey(entries[j])
	})
	got := make([]string, len(entries))
	for i := range entries {
		got[i] = entries[i].ID
	}
	want := "repo/alpha-a,repo/alpha-b,repo/zeta,repo/unsupported"
	if strings.Join(got, ",") != want {
		t.Fatalf("model catalog order = %v, want %s", got, want)
	}
}

func TestPairProjectorsMatchesSameRepositoryAndPrefersF16(t *testing.T) {
	textA := testEntry("ministral/ministral-Q4_K_M", "mistral3", false)
	textA.Repository = "ministral"
	textA.visionDimension = 3072
	textA.visionTemplateCapable = true
	mmprojF32 := testEntry("ministral/mmproj-F32", "clip", true)
	mmprojF32.Repository = "ministral"
	mmprojF32.FileName = "mmproj-F32.gguf"
	mmprojF32.visionDimension = 3072
	mmprojF32.visionPixtralProjector = true
	mmprojF16 := testEntry("ministral/mmproj-F16", "clip", true)
	mmprojF16.Repository = "ministral"
	mmprojF16.FileName = "mmproj-F16.gguf"
	mmprojF16.visionDimension = 3072
	mmprojF16.visionPixtralProjector = true
	// A text-only model in a different repository must not get paired.
	textB := testEntry("other/other-model", "llama", false)
	textB.Repository = "other"

	entries := []ModelEntry{textA, mmprojF32, mmprojF16, textB}
	pairProjectors(entries)

	if entries[0].ProjectorPath != mmprojF16.Path {
		t.Fatalf("ProjectorPath = %q, want the F16 variant %q", entries[0].ProjectorPath, mmprojF16.Path)
	}
	if entries[3].ProjectorPath != "" {
		t.Fatalf("cross-repository entry got a ProjectorPath: %q", entries[3].ProjectorPath)
	}
	// Projector entries themselves are never paired with anything.
	if entries[1].ProjectorPath != "" || entries[2].ProjectorPath != "" {
		t.Fatal("a projector entry should never itself get a ProjectorPath")
	}
}

func TestDiscoverModelsPairsOnlyRunnablePixtralVisionModels(t *testing.T) {
	root := t.TempDir()
	write := func(relative string, data []byte) string {
		t.Helper()
		path := filepath.Join(root, relative)
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		return path
	}

	goodTextPath := write(filepath.Join("safe", "model.gguf"), buildVisionCatalogTextGGUF(8, true))
	goodProjectorPath := write(filepath.Join("safe", "mmproj-F16.gguf"), buildVisionCatalogProjectorGGUF("pixtral", 8, true))
	// Neither a non-Pixtral encoder nor a matching filename with the wrong
	// decoder width is a usable companion. The latter has a better filename
	// rank than no candidate, so this also proves compatibility is checked
	// before the F16/BF16/F32 preference.
	write(filepath.Join("safe", "mmproj-Qwen-F16.gguf"), buildVisionCatalogProjectorGGUF("qwen2vl", 8, true))
	write(filepath.Join("safe", "mmproj-wrong-F16.gguf"), buildVisionCatalogProjectorGGUF("pixtral", 16, true))

	noTemplatePath := write(filepath.Join("no-mistral-markers", "model.gguf"), buildVisionCatalogTextGGUF(8, false))
	write(filepath.Join("no-mistral-markers", "mmproj-F16.gguf"), buildVisionCatalogProjectorGGUF("pixtral", 8, true))

	// These two directories deliberately share the same final name. Pairing
	// by ModelEntry.Repository alone would cross-wire them; exact parent paths
	// must keep each model isolated.
	firstTextPath := write(filepath.Join("first", "shared", "model.gguf"), buildVisionCatalogTextGGUF(8, true))
	write(filepath.Join("second", "shared", "mmproj-F16.gguf"), buildVisionCatalogProjectorGGUF("pixtral", 8, true))

	entries, err := DiscoverModels(root, nil)
	if err != nil {
		t.Fatal(err)
	}
	byPath := make(map[string]ModelEntry, len(entries))
	for _, entry := range entries {
		byPath[entry.Path] = entry
	}
	if got := byPath[goodTextPath].ProjectorPath; got != goodProjectorPath {
		t.Fatalf("safe ProjectorPath = %q, want %q", got, goodProjectorPath)
	}
	if got := byPath[noTemplatePath].ProjectorPath; got != "" {
		t.Fatalf("Mistral-token-incompatible ProjectorPath = %q, want empty", got)
	}
	if got := byPath[firstTextPath].ProjectorPath; got != "" {
		t.Fatalf("cross-directory ProjectorPath = %q, want empty", got)
	}
	if got, err := PairedVisionProjectorPath(goodTextPath); err != nil || got != goodProjectorPath {
		t.Fatalf("PairedVisionProjectorPath() = %q, %v; want %q, nil", got, err, goodProjectorPath)
	}
}

func buildVisionCatalogTextGGUF(dim uint32, imageTemplate bool) []byte {
	tokens := []any{"<unk>", "<s>", "</s>"}
	template := "{{ '[INST]' }}{{ '[IMG]' }}{{ '[/INST]' }}"
	if imageTemplate {
		tokens = append(tokens, "[INST]", "[/INST]", "[IMG]", "[IMG_BREAK]", "[IMG_END]")
	} else {
		template = "{{ '<|user|>' }}"
	}
	return buildGGUF(3, []ggufKV{
		{"general.architecture", ggufStr, "mistral3"},
		{"mistral3.embedding_length", ggufU32, dim},
		{"mistral3.context_length", ggufU32, uint32(128)},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, tokens}},
		{"tokenizer.chat_template", ggufStr, template},
	}, nil)
}

func buildVisionCatalogProjectorGGUF(projectorType string, projectionDim uint32, hasVision bool) []byte {
	return buildGGUF(3, []ggufKV{
		{"general.architecture", ggufStr, "clip"},
		{"clip.has_vision_encoder", ggufBool, hasVision},
		{"clip.projector_type", ggufStr, projectorType},
		{"clip.vision.projection_dim", ggufU32, projectionDim},
	}, nil)
}

func TestDiscoverModelsAndResolveModelPath(t *testing.T) {
	root := t.TempDir()
	modelPath := filepath.Join(root, "repo", "tiny.gguf")
	if err := os.MkdirAll(filepath.Dir(modelPath), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(modelPath, buildTinyLlamaGGUF(), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "broken.gguf"), []byte("not a GGUF"), 0o600); err != nil {
		t.Fatal(err)
	}

	var logs bytes.Buffer
	entries, err := DiscoverModels(root, &logs)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].ID != filepath.Join("repo", "tiny") || entries[0].ModelName != "tiny" || entries[0].ContextLength != 1024 || !entries[0].IsSupported {
		t.Fatalf("entries = %+v", entries)
	}
	if !strings.Contains(logs.String(), "Skipping") {
		t.Fatalf("expected broken model diagnostic, got %q", logs.String())
	}

	selector := "tiny"
	if got, err := ResolveModelPath(&selector, root); err != nil || got != modelPath {
		t.Fatalf("ResolveModelPath selector = %q, %v", got, err)
	}
	if got, err := ResolveModelPath(&modelPath, root); err != nil || got != modelPath {
		t.Fatalf("ResolveModelPath path = %q, %v", got, err)
	}
	if got, err := chooseFromDirectory(root, nil, nil, nil); err != nil || got != modelPath {
		t.Fatalf("chooseFromDirectory = %q, %v", got, err)
	}
}
