package cpuinfo

import (
	"runtime"
	"testing"
)

// PerformanceCores must either report a real asymmetric split or 0. Returning
// NumCPU would make threadCandidates add a duplicate, and returning something
// larger would let the autotuner oversubscribe.
func TestPerformanceCoresIsSaneOrZero(t *testing.T) {
	p := PerformanceCores()
	if p < 0 {
		t.Fatalf("PerformanceCores = %d, must not be negative", p)
	}
	if p > runtime.NumCPU() {
		t.Fatalf("PerformanceCores = %d exceeds NumCPU = %d", p, runtime.NumCPU())
	}
	if runtime.GOOS != "darwin" && p != 0 {
		t.Fatalf("PerformanceCores = %d on %s, want 0", p, runtime.GOOS)
	}
	t.Logf("GOOS=%s NumCPU=%d PerformanceCores=%d", runtime.GOOS, runtime.NumCPU(), p)
}
