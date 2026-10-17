package gopherllm

import (
	"reflect"
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

func TestThreadCandidatesAppleTopologies(t *testing.T) {
	for _, tc := range []struct {
		name                 string
		logical, performance int
		want                 []int
	}{
		{"M4 8-core", 8, 4, []int{8, 6, 4, 2}},
		{"M4 10-core", 10, 4, []int{10, 7, 5, 4, 2}},
		{"M4 Pro 12-core", 12, 8, []int{12, 10, 9, 8, 6, 3}},
		{"M4 Pro 14-core", 14, 10, []int{14, 12, 10, 7, 3}},
		{"unknown 14-core", 14, 0, []int{14, 10, 7, 3}},
		{"invalid P count", 8, 10, []int{8, 6, 4, 2}},
		{"symmetric", 8, 8, []int{8, 6, 4, 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := threadCandidatesForTopology(tc.logical, tc.performance); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("candidates = %v, want %v", got, tc.want)
			}
		})
	}
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
