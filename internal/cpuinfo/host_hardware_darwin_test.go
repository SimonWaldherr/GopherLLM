//go:build darwin && !ios

package cpuinfo

import (
	"fmt"
	"syscall"
	"testing"
)

func TestMacHardwareProbe(t *testing.T) {
	for _, tc := range []struct {
		chip                             string
		logical, performance, efficiency uint32
	}{
		{"Apple M4", 10, 4, 6},
		{"Apple M4 Pro", 12, 8, 4},
		{"Apple M4 Pro", 14, 10, 4},
	} {
		t.Run(fmt.Sprintf("%s/%d-core", tc.chip, tc.logical), func(t *testing.T) {
			text := map[string]string{"hw.model": "  test-mac\n", "machdep.cpu.brand_string": " " + tc.chip + " "}
			numbers := map[string]uint32{
				"hw.cpufamily": 0xf1234567, "hw.nperflevels": 2,
				"hw.perflevel0.logicalcpu": tc.performance,
				"hw.perflevel1.logicalcpu": tc.efficiency,
				"hw.optional.arm.FEAT_SME": 1, "hw.optional.arm.FEAT_SME2": 1,
			}
			got := readMacHardware(func(key string) (string, error) {
				value, ok := text[key]
				if !ok {
					t.Fatalf("unexpected string sysctl %q", key)
				}
				return value, nil
			}, func(key string) (uint32, error) {
				value, ok := numbers[key]
				if !ok {
					t.Fatalf("unexpected numeric sysctl %q", key)
				}
				return value, nil
			})
			want := macHardware{model: "test-mac", chip: tc.chip, family: 0xf1234567, performance: tc.performance, efficiency: tc.efficiency, sme: true, sme2: true}
			if got != want {
				t.Fatalf("probe = %+v, want %+v", got, want)
			}
		})
	}
}

func TestMacHardwareProbeUnavailable(t *testing.T) {
	for _, err := range []error{syscall.ENOENT, syscall.EPERM, syscall.EIO} {
		got := readMacHardware(func(string) (string, error) {
			return "must be ignored", err
		}, func(string) (uint32, error) {
			return 1, err
		})
		if got != (macHardware{}) || got.fingerprint() != "" {
			t.Fatalf("error %v produced hardware identity %+v", err, got)
		}
	}
}

func TestMacHardwareFingerprintSeparatesChipsAndModels(t *testing.T) {
	base := macHardware{model: "test-mac", chip: "Apple M4 Pro", family: 1, performance: 8, efficiency: 4, sme: true, sme2: true}
	otherChip, otherModel, otherBin, otherFeatures := base, base, base, base
	otherChip.chip = "Apple M2 Max" // The same 8+4 core count is insufficient.
	otherModel.model = "another-chassis"
	otherBin.performance = 10
	otherFeatures.sme, otherFeatures.sme2 = false, false
	seen := map[string]bool{}
	for _, hw := range []macHardware{base, otherChip, otherModel, otherBin, otherFeatures} {
		key := hw.fingerprint()
		if key == "" || seen[key] || hw.fingerprint() != key {
			t.Fatalf("missing, colliding or unstable hardware identity %q", key)
		}
		seen[key] = true
	}
}

func TestLocalMacHardwareSnapshot(t *testing.T) {
	if localMacHardware() != localMacHardware() {
		t.Fatal("local hardware snapshot changed")
	}
	t.Logf("local Mac: %+v", localMacHardware())
}
