package engine

import "unsafe"

// A loaded tensor may have been padded into Go-owned storage. Only a complete
// view of the original OS mapping may escape to Metal through bytesNoCopy.
func weightBytesInMapping(raw, mapping []byte) bool {
	if len(raw) == 0 || len(raw) > len(mapping) {
		return false
	}
	base := uintptr(unsafe.Pointer(unsafe.SliceData(mapping)))
	address := uintptr(unsafe.Pointer(unsafe.SliceData(raw)))
	return address >= base && address-base <= uintptr(len(mapping)-len(raw))
}
