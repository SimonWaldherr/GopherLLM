package gopherllm

import (
	"testing"
	"time"
)

func TestRunnerCloseWaitsForActiveVisionReader(t *testing.T) {
	r := &Runner{vision: &PixtralVisionWeights{}}
	r.visionMu.RLock()
	closed := make(chan struct{})
	go func() {
		_ = r.Close()
		close(closed)
	}()

	select {
	case <-closed:
		t.Fatal("Close released the vision tower while a reader held its lease")
	case <-time.After(20 * time.Millisecond):
	}

	r.visionMu.RUnlock()
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("Close did not finish after the vision reader released its lease")
	}
	if r.HasVision() {
		t.Fatal("Close retained the vision tower")
	}
}
