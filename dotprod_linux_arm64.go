//go:build linux && arm64

package gopherllm

import (
	"encoding/binary"
	"os"
)

// Linux (and Android, which builds with the linux tag) publishes the kernel's
// CPU feature mask through the ELF auxiliary vector. /proc/self/auxv is that
// vector verbatim: a NULL-terminated array of (tag, value) pairs, both native
// words, so 16 bytes per entry on arm64.
//
// Reading it avoids both cgo and golang.org/x/sys/cpu — this package carries no
// third-party dependencies — and avoids getauxval(3), which would need a libc
// call. The file is a few hundred bytes and read once at init.
const (
	// atHWCAP is AT_HWCAP from <elf.h>: the first arm64 feature word.
	atHWCAP = 16
	// atNull terminates the vector.
	atNull = 0
	// hwcapASIMDDP is HWCAP_ASIMDDP from the arm64 kernel uapi: FEAT_DotProd,
	// i.e. the SDOT/UDOT instructions q8k_dot_arm64.s emits as WORDs.
	hwcapASIMDDP = 1 << 20
)

func probeDotProd() bool {
	// auxv has no stat size on procfs; os.ReadFile grows its buffer to fit.
	data, err := os.ReadFile("/proc/self/auxv")
	if err != nil {
		// Locked-down container or a kernel without procfs. Assuming the
		// feature is absent costs speed; assuming it is present costs a
		// SIGILL, so this stays conservative.
		return false
	}
	// arm64 is little-endian in Go: GOARCH=arm64 is LE-only, there is no
	// arm64be port, so the word decode needs no endianness branch.
	for i := 0; i+16 <= len(data); i += 16 {
		tag := binary.LittleEndian.Uint64(data[i:])
		val := binary.LittleEndian.Uint64(data[i+8:])
		if tag == atNull {
			break
		}
		if tag == atHWCAP {
			return val&hwcapASIMDDP != 0
		}
	}
	return false
}
