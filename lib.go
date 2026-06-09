package main

import (
	"io"
	"os"
)

const Version = "0.3.0-go"

var errWriter io.Writer = os.Stderr

func stderr() io.Writer {
	if errWriter == nil {
		return os.Stderr
	}
	return errWriter
}
