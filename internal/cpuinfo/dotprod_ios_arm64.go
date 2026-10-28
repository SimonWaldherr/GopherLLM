//go:build ios && arm64

package cpuinfo

import "syscall"

// Dot-product instructions are optional on older iPhones. Query Apple's
// public CPU feature sysctl before enabling any SDOT kernel; never probe by
// executing an instruction. Missing or inaccessible keys keep the fallback.
func probeDotProd() bool {
	return probeDotProdWithSysctl(syscall.SysctlUint32)
}
