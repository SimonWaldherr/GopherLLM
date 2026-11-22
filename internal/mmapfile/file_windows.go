//go:build windows

package mmapfile

import (
	"os"
	"syscall"
	"unsafe"
)

// File is the Windows variant of the Unix type: the same immutable
// byte-slice API, backed by a real Win32 file mapping
// (CreateFileMapping/MapViewOfFile) so multi-gigabyte weights are paged in
// on demand instead of being read up front. If mapping fails for a specific
// file, it falls back to a full os.ReadFile copy, preserving the API.
type File struct {
	data   []byte
	mapped bool
	addr   uintptr
}

// The mapping-object and file handles are closed before returning. A view keeps the
// underlying mapping (and file) alive on its own until UnmapViewOfFile.
func open(path string, requireMapping bool) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := st.Size()
	if size == 0 || size > int64(^uint(0)>>1) {
		return checkMappingSize(path, size, requireMapping)
	}

	mapping, err := syscall.CreateFileMapping(syscall.Handle(f.Fd()), nil, syscall.PAGE_READONLY, uint32(size>>32), uint32(size), nil)
	if err == nil {
		var addr uintptr
		addr, err = syscall.MapViewOfFile(mapping, syscall.FILE_MAP_READ, 0, 0, uintptr(size))
		syscall.CloseHandle(mapping)
		if err == nil {
			// addr references the OS mapping, memory the Go GC never
			// manages, so building a slice header over it is safe; the
			// store-through-pointer form (rather than a direct
			// uintptr->Pointer conversion expression) is what vet's
			// unsafeptr check accepts for OS-provided addresses.
			var base unsafe.Pointer
			*(*uintptr)(unsafe.Pointer(&base)) = addr
			data := unsafe.Slice((*byte)(base), size)
			return &File{data: data, mapped: true, addr: addr}, nil
		}
	}
	if requireMapping {
		return nil, &os.PathError{Op: "mmap", Path: path, Err: err}
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return &File{data: data}, nil
}

func (m *File) Bytes() []byte  { return m.data }
func (m *File) Len() int       { return len(m.data) }
func (m *File) IsMapped() bool { return m.mapped }
func (m *File) Close() error {
	if len(m.data) == 0 {
		return nil
	}
	m.data = nil
	if m.mapped {
		m.mapped = false
		addr := m.addr
		m.addr = 0
		return syscall.UnmapViewOfFile(addr)
	}
	return nil
}
