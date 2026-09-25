package huggingface

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestLayaBundleFiles(t *testing.T) {
	files, e := layaFiles("multilingual")
	if e != nil {
		t.Fatal(e)
	}
	if len(files) != 5 || files[0] != "multilingual/rl_agent_config.json" || files[4] != "multilingual/model.safetensors" {
		t.Fatal(files)
	}
	for _, s := range []string{"../escape", "/absolute", "a/../b", "a\\b", ".", "a/"} {
		if _, e := layaFiles(s); e == nil {
			t.Errorf("accepted %q", s)
		}
	}
}
func TestDownloadLayaOfflineMiss(t *testing.T) {
	t.Setenv("HF_HOME", t.TempDir())
	if _, e := DownloadLaya(context.Background(), "convaiinnovations/laya", "", nil, Options{Offline: true}); e == nil {
		t.Fatal("uncached offline bundle succeeded")
	}
}
func TestDownloadLayaCached(t *testing.T) {
	// Optional real cache check after the CLI's end-to-end download smoke test.
	if os.Getenv("GOPHERLLM_LAYA_DOWNLOAD_TEST") != "1" {
		t.Skip("requires a populated HF cache")
	}
	dir, e := DownloadLaya(context.Background(), "convaiinnovations/laya", "multilingual", nil, Options{Offline: true})
	if e != nil {
		t.Fatal(e)
	}
	for _, name := range []string{"rl_agent_config.json", "model.safetensors", "tokenizer/tokenizer.json", "encoder/config.json"} {
		if _, e = os.Stat(filepath.Join(dir, name)); e != nil {
			t.Fatal(e)
		}
	}
}
