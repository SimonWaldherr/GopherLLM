package gopherllm

import (
	"strings"
	"testing"
)

func TestSupportedArchitecturesMatchesArchitectureSupported(t *testing.T) {
	list := SupportedArchitectures()
	if len(list) == 0 {
		t.Fatal("SupportedArchitectures returned nothing")
	}
	for _, arch := range list {
		if !ArchitectureSupported(arch) {
			t.Fatalf("SupportedArchitectures lists %q but ArchitectureSupported(%q) = false", arch, arch)
		}
	}
	// Every arch TestArchitectureSupportedCoversImplementedLoaders checks
	// true for must also appear in the list, or the two would have drifted
	// apart despite sharing the same backing data.
	for _, arch := range []string{"llama", "qwen3", "gemma3", "bert", "phi2", "olmo2"} {
		found := false
		for _, a := range list {
			if a == arch {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("SupportedArchitectures is missing %q", arch)
		}
	}
}

func TestSupportedArchitecturesReturnsAFreshCopy(t *testing.T) {
	a := SupportedArchitectures()
	a[0] = "mutated"
	b := SupportedArchitectures()
	if b[0] == "mutated" {
		t.Fatal("SupportedArchitectures shares backing storage across calls; mutation leaked")
	}
}

func TestUnsupportedArchitectureErrorPointsToSupportedArchitectures(t *testing.T) {
	g := metaOnlyGGUF(t, []ggufKV{
		{"general.architecture", ggufStr, "olmoe"},
		{"olmoe.block_count", ggufU32, uint32(2)},
		{"olmoe.embedding_length", ggufU32, uint32(8)},
	})
	_, err := runnerFromParsedGGUF(nil, g, false, LoadOptions{})
	if err == nil || !strings.Contains(err.Error(), "SupportedArchitectures") {
		t.Fatalf("error = %v, want a pointer to SupportedArchitectures", err)
	}
}

func TestLooksLikeHubReference(t *testing.T) {
	cases := []struct {
		path string
		want bool
	}{
		{"bartowski/Llama-3.2-1B-Instruct-GGUF", true},
		{"owner/repo", true},
		{"model.gguf", false},
		{"./model.gguf", false},
		{"/home/user/models/model.gguf", false},
		{`C:\models\model.gguf`, false},
		{"owner/repo/extra/path.gguf", false},
		{"owner/repo/nested", false},
		{"/owner/repo", false},
		{"", false},
	}
	for _, c := range cases {
		if got := looksLikeHubReference(c.path); got != c.want {
			t.Errorf("looksLikeHubReference(%q) = %v, want %v", c.path, got, c.want)
		}
	}
}

func TestRunnerFromPathHubReferenceErrorMentionsHuggingFace(t *testing.T) {
	_, _, err := RunnerFromPath("bartowski/Llama-3.2-1B-Instruct-GGUF")
	if err == nil {
		t.Fatal("expected an error for a nonexistent path")
	}
	if !strings.Contains(err.Error(), "huggingface") {
		t.Fatalf("error = %v, want a pointer to the huggingface package", err)
	}
}

func TestRunnerFromPathLocalTypoErrorStaysPlain(t *testing.T) {
	_, _, err := RunnerFromPath("/tmp/definitely-not-a-real-model.gguf")
	if err == nil {
		t.Fatal("expected an error for a nonexistent path")
	}
	if strings.Contains(err.Error(), "huggingface") {
		t.Fatalf("error = %v, a plain local .gguf path must not get the Hub-reference hint", err)
	}
}
