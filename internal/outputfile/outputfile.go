// Package outputfile writes generated CLI output without replacing existing files.
package outputfile

import (
	"fmt"
	"io"
	"os"
)

// CheckNew rejects an existing destination before expensive generation starts.
// Write checks again when creating the file, so a later arrival is not replaced.
func CheckNew(path string) error {
	if _, err := os.Lstat(path); err == nil {
		return fmt.Errorf("output already exists: %s", path)
	} else if !os.IsNotExist(err) {
		return err
	}
	return nil
}

// Write creates path exclusively and removes incomplete output if encoding or
// closing fails. encode must finish writing before it returns.
func Write(path string, encode func(io.Writer) error) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	complete := false
	defer func() {
		_ = f.Close()
		if !complete {
			_ = os.Remove(path)
		}
	}()
	if err := encode(f); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	complete = true
	return nil
}
