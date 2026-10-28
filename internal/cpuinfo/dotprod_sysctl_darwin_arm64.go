//go:build darwin && arm64

package cpuinfo

// Apple's ARM feature sysctls report supported instructions, including on
// iOS. The public iPhoneOS SDK exposes the matching FEAT_DotProd capability;
// XNU exports this read-only integer without a macOS-only restriction:
// https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_mib.c
// https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/arm/arm_features.inc
// Keep the reader explicit so missing/denied feature queries can be tested on
// a Mac without executing optional instructions or changing global state.
func probeDotProdWithSysctl(read func(string) (uint32, error)) bool {
	value, err := read("hw.optional.arm.FEAT_DotProd")
	return err == nil && value != 0
}
