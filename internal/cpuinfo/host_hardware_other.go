//go:build !darwin || ios

package cpuinfo

// HardwareFingerprint returns an empty string on platforms without a hardware probe.
func HardwareFingerprint() string { return "" }
