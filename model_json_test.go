package gopherllm

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLayaConfigReadLimits(t *testing.T) {
	const limit = 1 << 20
	configs := map[string]string{
		"rl_agent_config.json": `{}`,
		"encoder/config.json":  `{"model_type":"modernbert","hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"vocab_size":4}`,
	}
	for _, filename := range []string{"rl_agent_config.json", "encoder/config.json"} {
		t.Run(filename, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.Mkdir(filepath.Join(dir, "encoder"), 0700); err != nil {
				t.Fatal(err)
			}
			for name, data := range configs {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0600); err != nil {
					t.Fatal(err)
				}
			}
			for _, size := range []int{limit, limit + 1} {
				data := configs[filename] + strings.Repeat(" ", size-len(configs[filename]))
				if err := os.WriteFile(filepath.Join(dir, filename), []byte(data), 0600); err != nil {
					t.Fatal(err)
				}
				_, err := OpenLaya(context.Background(), dir)
				if size == limit {
					// A valid configuration reaches the intentionally absent tokenizer.
					var pathErr *os.PathError
					if !errors.Is(err, os.ErrNotExist) || !errors.As(err, &pathErr) || pathErr.Path != filepath.Join(dir, "tokenizer/tokenizer.json") {
						t.Fatalf("configuration at limit was rejected: %v", err)
					}
				} else if err == nil || !strings.Contains(err.Error(), "no larger than") {
					t.Fatalf("oversized configuration was not rejected before loading: %v", err)
				}
			}
		})
	}
}

func TestGenerationMetadataLimits(t *testing.T) {
	for _, tc := range []struct {
		name  string
		limit int
		valid string
		read  func(string) error
	}{
		{"tts", 1 << 20, `{}`, func(path string) error {
			var config ttsVITSConfig
			return ttsReadJSON(path, &config)
		}},
		{"diffusion", 64 << 10, diffusionCIFARConfig, func(path string) error {
			return diffusionValidateJSON(path, diffusionCIFARConfig, diffusionConfigDefaults)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.json")
			if err := tc.read(path); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("missing file: %v", err)
			}
			for _, sample := range []struct {
				name, data string
				valid      bool
			}{
				{"at limit", tc.valid + strings.Repeat(" ", tc.limit-len(tc.valid)), true},
				{"above limit", tc.valid + strings.Repeat(" ", tc.limit+1-len(tc.valid)), false},
				{"second document", tc.valid + `{}`, false},
				{"empty", "", false},
			} {
				t.Run(sample.name, func(t *testing.T) {
					if err := os.WriteFile(path, []byte(sample.data), 0600); err != nil {
						t.Fatal(err)
					}
					if err := tc.read(path); (err == nil) != sample.valid {
						t.Fatalf("valid=%v: %v", sample.valid, err)
					}
				})
			}
			if err := tc.read(filepath.Dir(path)); err == nil {
				t.Fatal("accepted directory as metadata")
			}
		})
	}
}
