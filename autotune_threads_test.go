package gopherllm

import (
	"runtime"
	"strconv"
	"testing"
)

// Thread counts supplied by an embedding application need not coincide with
// the topology candidates. Equal measurements must preserve that actual
// incumbent, rather than silently replacing it with the full CPU count.
func TestTuneThreadsPreservesExplicitIncumbent(t *testing.T) {
	before := captureTunerConfig(Config{})
	previousProcs := runtime.GOMAXPROCS(0)
	defer func() {
		before.apply()
		runtime.GOMAXPROCS(previousProcs)
	}()

	nproc := runtime.NumCPU()
	candidates := threadCandidates(nproc)
	type testCase struct {
		name    string
		current int
	}
	cases := []testCase{
		{"above_cpu_count", nproc + 1},
	}
	if nproc > 1 {
		cases = append(cases, testCase{"existing_candidate", candidates[0]})
	}
	for current := nproc - 1; current >= 1; current-- {
		found := false
		for _, candidate := range candidates {
			if candidate == current {
				found = true
				break
			}
		}
		if !found {
			cases = append(cases, testCase{"omitted_below_cpu_count", current})
			break
		}
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			SetNumThreads(tc.current)
			runtime.GOMAXPROCS(tc.current)
			tuner := &autoTuner{opts: AutoTuneOptions{Rounds: 2}.withDefaults()}
			tuner.decodeProbeFn = func() float64 { return 100 }
			tuner.tuneThreads()

			if got := numThreads(); got != tc.current {
				t.Fatalf("equal measurements changed threads from %d to %d", tc.current, got)
			}
			if got := runtime.GOMAXPROCS(0); got != tc.current {
				t.Fatalf("GOMAXPROCS = %d, want incumbent %d", got, tc.current)
			}
			if len(tuner.trials) != 1 {
				t.Fatalf("thread trials = %d, want 1", len(tuner.trials))
			}
			trial := tuner.trials[0]
			want := strconv.Itoa(tc.current)
			if trial.Incumbent != want || trial.Chosen != want {
				t.Fatalf("thread trial lost incumbent %s: %+v", want, trial)
			}
			seen := make(map[int]bool)
			previous := 0
			for i, sample := range trial.Candidates {
				n, err := strconv.Atoi(sample.Value)
				if err != nil || n < 1 {
					t.Fatalf("invalid thread candidate %q", sample.Value)
				}
				if seen[n] || i > 0 && previous < n {
					t.Fatalf("thread candidates must be unique and descending: %+v", trial.Candidates)
				}
				seen[n] = true
				previous = n
			}
			if !seen[tc.current] {
				t.Fatalf("current thread count %d was not measured: %+v", tc.current, trial.Candidates)
			}
		})
	}
}
