package gopherllm

import (
	"runtime"
	"testing"
)

// performanceCores must either report a real asymmetric split or 0. Returning
// NumCPU would make threadCandidates add a duplicate, and returning something
// larger would let the autotuner oversubscribe.
func TestPerformanceCoresIsSaneOrZero(t *testing.T) {
	p := performanceCores()
	if p < 0 {
		t.Fatalf("performanceCores = %d, must not be negative", p)
	}
	if p > runtime.NumCPU() {
		t.Fatalf("performanceCores = %d exceeds NumCPU = %d", p, runtime.NumCPU())
	}
	if runtime.GOOS != "darwin" && p != 0 {
		t.Fatalf("performanceCores = %d on %s, want 0", p, runtime.GOOS)
	}
	t.Logf("GOOS=%s NumCPU=%d performanceCores=%d", runtime.GOOS, runtime.NumCPU(), p)
}

// The candidate set must stay sorted descending, deduplicated, and in range no
// matter what performanceCores reports — including the M2 Max shape (12 logical
// CPUs, 8 performance cores) where the plain fractions {12, 9, 6, 3} miss 8.
func TestThreadCandidatesShape(t *testing.T) {
	for _, nproc := range []int{1, 2, 4, 8, 10, 12, 16, 20, 24} {
		got := threadCandidates(nproc)
		if len(got) == 0 {
			t.Fatalf("nproc=%d: no candidates", nproc)
		}
		seen := map[int]bool{}
		for i, n := range got {
			if n < 1 || n > nproc {
				t.Fatalf("nproc=%d: candidate %d out of range in %v", nproc, n, got)
			}
			if seen[n] {
				t.Fatalf("nproc=%d: duplicate %d in %v", nproc, n, got)
			}
			seen[n] = true
			if i > 0 && got[i-1] < n {
				t.Fatalf("nproc=%d: not descending: %v", nproc, got)
			}
		}
		if !seen[nproc] {
			t.Errorf("nproc=%d: the all-cores candidate is missing from %v", nproc, got)
		}
	}
}
