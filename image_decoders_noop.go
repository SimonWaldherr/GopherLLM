//go:build noimagedecoders

package gopherllm

// This file is the opt-out half of the split described in image_decoders.go:
// under -tags noimagedecoders neither image/jpeg nor image/png is imported,
// which removes seven standard-library packages (compress/zlib,
// compress/flate, hash/adler32 and hash/crc32 among them) from the
// dependency closure of every program that embeds this library. The tag is
// there for exactly that audience — text-only consumers, and vendoring or
// supply-chain reviews where the closure is the thing being counted.
//
// The price is real and is paid at runtime, not at compile time: nothing in
// the public API disappears, but ImageContent bytes can no longer be turned
// into an image.Image, so the vision path stops accepting PNG and JPEG
// input. Everything text-only is untouched. `image` itself stays imported by
// image_preprocess.go because image.Image is part of this package's exported
// signatures; removing it would be a breaking change in exchange for two
// packages, which is not a trade worth offering.
//
// This is not all-or-nothing for the consumer either. Go's image format
// registry (image.RegisterFormat) is process-global and is populated from
// package init functions, so a program built with this tag can re-enable
// precisely the formats it wants with its own blank import — `import _
// "image/png"` to take PNG back without paying for JPEG, or a decoder for
// some format this module would never depend on itself. DecodeImageBytes
// therefore calls image.Decode unconditionally in both builds, so that those
// foreign registrations keep working and only the error message differs.
const imageDecodersLinked = false
