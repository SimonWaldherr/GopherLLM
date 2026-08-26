package gopherllm

import (
	"errors"
	"sync"
	"testing"
	"time"
)

func TestRunnerResourceOperationsRejectAfterClose(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	cacheDir := t.TempDir()
	t.Setenv("LOCALAPPDATA", cacheDir) // os.UserCacheDir on Windows
	t.Setenv("XDG_CACHE_HOME", cacheDir)
	t.Setenv("HOME", cacheDir)
	key := r.autoTuneKey()
	if err := SaveAutoTune(AutoTuneResult{Version: 1, Key: key, Threads: 1}); err != nil {
		t.Fatalf("SaveAutoTune: %v", err)
	}
	if _, ok := r.LoadAutoTune(); !ok {
		t.Fatal("LoadAutoTune did not read the valid pre-close cache entry")
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}

	if _, err := r.Generate("hello", DefaultGenerationOptions()); !errors.Is(err, ErrRunnerClosed) {
		t.Fatalf("Generate error = %v, want ErrRunnerClosed", err)
	}
	if _, err := r.Embed("hello"); !errors.Is(err, ErrRunnerClosed) {
		t.Fatalf("Embed error = %v, want ErrRunnerClosed", err)
	}
	if _, err := r.NearestTokens(0, 1); !errors.Is(err, ErrRunnerClosed) {
		t.Fatalf("NearestTokens error = %v, want ErrRunnerClosed", err)
	}
	if _, err := r.AutoTune(AutoTuneOptions{}); !errors.Is(err, ErrRunnerClosed) {
		t.Fatalf("AutoTune error = %v, want ErrRunnerClosed", err)
	}
	if _, ok := r.LoadAutoTune(); ok {
		t.Fatal("LoadAutoTune reported a cached result for a closed Runner")
	}
	if err := RunKernelBench(r, "", 1, 0, true); !errors.Is(err, ErrRunnerClosed) {
		t.Fatalf("RunKernelBench error = %v, want ErrRunnerClosed", err)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("second Close = %v, want nil", err)
	}
}

func TestRunnerCloseWaitsForActiveGeneration(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}

	opts := DefaultGenerationOptions()
	opts.SystemPrompt = ""
	opts.MaxTokens = 1
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1

	started := make(chan struct{})
	release := make(chan struct{})
	generated := make(chan error, 1)
	var once sync.Once
	go func() {
		_, err := r.GenerateStream("hello", opts, func(string) {
			once.Do(func() {
				close(started)
				<-release
			})
		})
		generated <- err
	}()

	select {
	case <-started:
	case <-time.After(time.Second):
		close(release)
		t.Fatal("generation never reached the streaming callback")
	}

	closed := make(chan error, 1)
	go func() { closed <- r.Close() }()
	select {
	case err := <-closed:
		t.Fatalf("Close returned while generation was active: %v", err)
	case <-time.After(20 * time.Millisecond):
	}

	close(release)
	select {
	case err := <-generated:
		if err != nil {
			t.Fatalf("GenerateStream: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("generation did not finish after callback release")
	}
	select {
	case err := <-closed:
		if err != nil {
			t.Fatalf("Close: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Close did not finish after generation released its resources")
	}
}
