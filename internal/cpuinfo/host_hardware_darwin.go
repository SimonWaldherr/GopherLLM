//go:build darwin && !ios

package cpuinfo

import (
	"fmt"
	"strings"
	"sync"
	"syscall"
)

// A CPU count is not a hardware identity: M2 Max, M3 Pro and M4 Pro can all
// have twelve logical CPUs, with different core mixes and memory systems.
// Include the Mac model as well, since a fanless MacBook Air and a MacBook Pro
// need not sustain the same settings even when they use the same chip.
type macHardware struct {
	model, chip             string
	family                  uint32
	performance, efficiency uint32
	sme, sme2               bool
}

func readMacHardware(text func(string) (string, error), number func(string) (uint32, error)) macHardware {
	readText := func(name string) string {
		value, err := text(name)
		if err != nil {
			return ""
		}
		return strings.TrimSpace(value)
	}
	readNumber := func(name string) uint32 {
		value, err := number(name)
		if err != nil {
			return 0
		}
		return value
	}
	h := macHardware{
		model:  readText("hw.model"),
		chip:   readText("machdep.cpu.brand_string"),
		family: readNumber("hw.cpufamily"),
		sme:    readNumber("hw.optional.arm.FEAT_SME") != 0,
		sme2:   readNumber("hw.optional.arm.FEAT_SME2") != 0,
	}
	if readNumber("hw.nperflevels") >= 2 {
		h.performance = readNumber("hw.perflevel0.logicalcpu")
		h.efficiency = readNumber("hw.perflevel1.logicalcpu")
	}
	return h
}

func (h macHardware) fingerprint() string {
	if h == (macHardware{}) {
		// Missing or denied sysctls keep the existing portable identity.
		return ""
	}
	return fmt.Sprintf("mac=%q/chip=%q/family=%08x/p%d/e%d/sme=%t/sme2=%t",
		h.model, h.chip, h.family, h.performance, h.efficiency, h.sme, h.sme2)
}

// Read immutable hardware properties once, without cgo or optional ISA
// execution. Feature bits distinguish OS-exposed capabilities only; Accelerate
// selects its own matrix implementation at runtime.
var localMacHardware = sync.OnceValue(func() macHardware {
	return readMacHardware(syscall.Sysctl, syscall.SysctlUint32)
})

// HardwareFingerprint identifies the host for saved tuning results, or returns
// an empty string when hardware details are unavailable.
func HardwareFingerprint() string { return localMacHardware().fingerprint() }
