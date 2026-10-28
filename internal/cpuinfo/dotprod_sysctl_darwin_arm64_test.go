//go:build darwin && arm64

package cpuinfo

import (
	"syscall"
	"testing"
)

func TestDotProdSysctlFeatureGate(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value uint32
		err   error
		want  bool
	}{
		{name: "supported", value: 1, want: true},
		{name: "unsupported"},
		{name: "missing", err: syscall.ENOENT},
		{name: "denied", err: syscall.EPERM},
		{name: "invalid size", err: syscall.EIO},
		{name: "error with nonzero value", value: 1, err: syscall.EIO},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			got := probeDotProdWithSysctl(func(name string) (uint32, error) {
				calls++
				if name != "hw.optional.arm.FEAT_DotProd" {
					t.Fatalf("queried %q instead of the documented DotProd feature", name)
				}
				return tc.value, tc.err
			})
			if got != tc.want || calls != 1 {
				t.Fatalf("feature gate = %v after %d queries, want %v after one", got, calls, tc.want)
			}
		})
	}
}

func TestDotProdSysctlHonorsDisableSIMD(t *testing.T) {
	t.Setenv("GOPHERLLM_DISABLE_SIMD", "1")
	if HasDotProd() {
		t.Fatal("SIMD disable must override hardware feature detection")
	}
}
