package huggingface

import (
	"context"
	"fmt"
	"io"
	"path"
	"path/filepath"
	"strings"
)

// DownloadLaya downloads a complete native Laya checkpoint into the shared
// Hugging Face snapshot cache and returns its local directory. It supports any
// repository with Laya's ModernBERT/mmBERT decision graph, including fine-tunes.
// subfolder selects a bundled checkpoint (e.g. multilingual or typed-decisions).
// ref supports owner/repo[@revision]; opts retains offline and progress behavior.
func DownloadLaya(ctx context.Context, ref, subfolder string, logw io.Writer, opts Options) (string, error) {
	files, err := layaFiles(subfolder)
	if err != nil {
		return "", err
	}
	paths, err := DownloadFiles(ctx, ref, files, logw, opts)
	if err != nil {
		return "", fmt.Errorf("download Laya: %w", err)
	}
	return filepath.Dir(paths[0]), nil
}
func layaFiles(subfolder string) ([]string, error) {
	if subfolder != "" && (strings.Contains(subfolder, "\\") || strings.HasPrefix(subfolder, "/") || path.Clean(subfolder) != subfolder || subfolder == "." || subfolder == ".." || strings.HasPrefix(subfolder, "../")) {
		return nil, fmt.Errorf("invalid Laya subfolder %q", subfolder)
	}
	names := []string{"rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "model.safetensors"}
	for i := range names {
		names[i] = path.Join(subfolder, names[i])
	}
	return names, nil
}
