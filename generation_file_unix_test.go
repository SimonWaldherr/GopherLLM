//go:build darwin || linux || freebsd || netbsd || openbsd

package gopherllm

import (
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

func TestGenerationRejectsFIFOConfigBeforeOpen(t *testing.T) {
	for _, tc := range []struct {
		name string
		read func(string) error
	}{
		{"diffusion", func(path string) error {
			return diffusionValidateJSON(path, diffusionCIFARConfig, diffusionConfigDefaults)
		}},
		{"tts", func(path string) error {
			var config map[string]any
			return ttsReadJSON(path, &config)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.json")
			if err := syscall.Mkfifo(path, 0600); err != nil {
				t.Fatal(err)
			}
			done := make(chan error, 1)
			go func() { done <- tc.read(path) }()
			select {
			case err := <-done:
				if err == nil {
					t.Fatal("accepted a FIFO as model configuration")
				}
			case <-time.After(time.Second):
				// Release a blocked reader before failing if the pre-open check
				// is removed. Nonblocking RDWR never waits for a peer.
				f, err := os.OpenFile(path, os.O_RDWR|syscall.O_NONBLOCK, 0600)
				if err == nil {
					_ = f.Close()
				}
				t.Fatal("configuration open blocked on a FIFO")
			}
		})
	}
}
