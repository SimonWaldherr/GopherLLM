//go:build !noimagedecoders

package gopherllm

// These two blank imports are the default half of the noimagedecoders split,
// and they are the reason this file exists at all: they are the single
// largest chunk of standard-library dependency closure the package imposes
// on consumers that never look at a picture. Measured with `go list -deps`,
// image/jpeg costs two extra packages and image/png costs five (it drags in
// compress/zlib, compress/flate, hash/adler32 and hash/crc32), so a
// text-only program that embeds this library links seven packages it can
// never reach. Building with -tags noimagedecoders drops all seven; this
// file is the default, so nothing changes for anyone who does not ask for
// it, and the vision path keeps accepting PNG and JPEG bytes as before.
//
// The `image` package itself is imported unconditionally by
// image_preprocess.go and always will be. It costs only two packages, and
// image.Image appears in the exported signatures of DecodeImageBytes,
// PreprocessImagePixtral and PreprocessImagePixtralDynamic — dropping it
// would be an API break rather than a build-time knob. See
// image_decoders_noop.go for what the tagged build gives up and how a
// consumer buys individual formats back.
import (
	_ "image/jpeg"
	_ "image/png"
)

// imageDecodersLinked tells DecodeImageBytes whether a failing image.Decode
// means "these bytes are not a picture" or "this binary has no decoder for
// them", which are the same opaque stdlib error otherwise. A constant rather
// than a variable so the compiler folds the branch away in whichever of the
// two builds it does not apply to; the paired hasQuantSIMD constants in
// kernels_*.go use the same pattern.
const imageDecodersLinked = true
