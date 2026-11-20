// Command gopherllm-convert-littlebit converts an already trained LittleBit
// checkpoint into a GopherLLM GGUF.
package main

import (
	"flag"
	"fmt"
	"os"
)

func main() {
	in := flag.String("in", "", "Hugging Face checkpoint directory")
	out := flag.String("out", "", "new GGUF file to create")
	variant := flag.Uint("variant", 0, "LittleBit variant (1 or 2); infer from use_itq by default")
	flag.Parse()
	if *in == "" || *out == "" || *variant > 2 || flag.NArg() > 0 {
		flag.Usage()
		os.Exit(2)
	}
	if err := convert(*in, *out, uint32(*variant)); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("Wrote %s\n", *out)
}
