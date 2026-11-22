package mmapfile

import (
	"errors"
	"os"
)

var errEmptyMapping = errors.New("cannot map an empty file")
var errMappingTooLarge = errors.New("file exceeds the addressable mapping size")

// Open maps path read-only, falling back to a full file read if the OS
// cannot map it. An empty file is exposed as an empty slice.
func Open(path string) (*File, error) {
	return open(path, false)
}

// OpenMapped requires an OS mapping. It never reads the entire file into
// the Go heap when mmap fails, even for a multi-gigabyte model shard.
func OpenMapped(path string) (*File, error) {
	return open(path, true)
}

func checkMappingSize(path string, size int64, requireMapping bool) (*File, error) {
	if size > int64(^uint(0)>>1) {
		return nil, &os.PathError{Op: "mmap", Path: path, Err: errMappingTooLarge}
	}
	if requireMapping {
		return nil, &os.PathError{Op: "mmap", Path: path, Err: errEmptyMapping}
	}
	return &File{}, nil
}
