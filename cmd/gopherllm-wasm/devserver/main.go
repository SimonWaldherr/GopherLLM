// Command devserver is a tiny static file server for manually exercising the
// wasm browser harness (cmd/gopherllm-wasm/testdata/harness) during
// development. It is a host-run tool, not part of the wasm build, and uses
// only net/http from the standard library.
package main

import (
	"flag"
	"log"
	"net/http"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:8090", "address to listen on")
	harnessDir := flag.String("harness", "cmd/gopherllm-wasm/testdata/harness", "directory serving index.html/app.js at /")
	binDir := flag.String("bin", "bin", "directory serving gopherllm.wasm/wasm_exec.js at /bin/")
	modelsDir := flag.String("models", "", "optional directory serving real (large) .gguf files at /models/, e.g. the repo root -- avoids copying a multi-GB model into the harness directory")
	flag.Parse()

	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.Dir(*harnessDir)))
	mux.Handle("/bin/", http.StripPrefix("/bin/", http.FileServer(http.Dir(*binDir))))
	if *modelsDir != "" {
		mux.Handle("/models/", http.StripPrefix("/models/", http.FileServer(http.Dir(*modelsDir))))
	}

	log.Printf("serving %s at http://%s/ and %s at http://%s/bin/", *harnessDir, *addr, *binDir, *addr)
	if *modelsDir != "" {
		log.Printf("serving %s at http://%s/models/", *modelsDir, *addr)
	}
	log.Fatal(http.ListenAndServe(*addr, mux))
}
