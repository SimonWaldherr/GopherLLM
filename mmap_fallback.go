//go:build windows

package gopherllm

import "os"

// MmapFile is the Windows variant of the mmap.go type: same immutable
// byte-slice API, but backed by a full os.ReadFile copy since this build does
// not use the Win32 mapping APIs. Weights borrowed from Bytes() remain valid
// for the Runner's lifetime just as in the mmap-backed variant.
type MmapFile struct {
	data []byte
}

// OpenMmap reads the whole file into memory (see type comment).
func OpenMmap(path string) (*MmapFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return &MmapFile{data: data}, nil
}

func (m *MmapFile) Bytes() []byte { return m.data }
func (m *MmapFile) Len() int      { return len(m.data) }
func (m *MmapFile) Close() error {
	m.data = nil
	return nil
}
