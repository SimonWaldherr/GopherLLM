//go:build darwin

package gopherllm

import "syscall"

// performanceCores returns the number of performance ("P") cores, or 0 when
// the system is homogeneous or the count cannot be determined.
//
// Apple Silicon is asymmetric: an M2 Max reports 12 logical CPUs, but only 8 of
// them are performance cores — the other 4 are efficiency cores that run the
// same matvec several times slower. For a memory-bound decode step, spreading
// work across all 12 means the 8 fast cores finish their share and then wait on
// the 4 slow ones at the next barrier, so the slowest core sets the pace.
//
// macOS exposes the core layout through the hw.perflevel<N> sysctl family,
// where perflevel0 is the fastest tier. This is read through the stdlib syscall
// package rather than cgo, so the cgo-free build keeps working.
//
// On homogeneous machines (all Intel Macs, and any Apple part where the sysctl
// is absent) hw.perflevel0.logicalcpu either equals the total or is missing; in
// both cases returning 0 tells the caller there is nothing special to do.
func performanceCores() int {
	n, err := syscall.SysctlUint32("hw.perflevel0.logicalcpu")
	if err != nil || n == 0 {
		return 0
	}
	// A single perflevel means the machine is homogeneous, which the caller
	// signals to itself by getting 0 back rather than a count equal to NumCPU.
	if levels, err := syscall.SysctlUint32("hw.nperflevels"); err == nil && levels < 2 {
		return 0
	}
	return int(n)
}
