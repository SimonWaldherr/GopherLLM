# SME int8 tile experiment

This package is separate from inference and changes no model dispatch. It tests
whether packing quantized weights and activations for a streaming matrix tile
is worth pursuing on an SME-capable Apple CPU. There is no measured SME speedup
on the development M2 Max: its hardware path is skipped.

The C microkernel uses **SME1 signed-int8 SMOPA**, accumulating exact int32 dot
products in ZA0. It also reports SME2 availability; it contains no SME2-specific
kernel. It does not implement Q4/Q6 decoding, block scales, floating-point
accumulation, whole-matrix scheduling or model prefill. Those costs would need
separate parity tests and complete projection/model benchmarks before any
production integration. The scalar benchmark is a correctness/cost reference,
not the existing optimized NEON kernel.

## Packing and ABI

Both operands start as `[rows][depth]`. Packing produces
`[ceil(depth/4)][streaming_int32_lanes][4]`: each lane contains four adjacent
int8 values along the dot-product depth. Missing rows and depth tails are zero.
The result is an ordinary row-major `[left_rows][right_rows]` int32 matrix.
Depth is capped at 131071 so full-range int8 input cannot overflow int32.

Tile dimensions come from `svcntsw()` at runtime, with predicates for partial
rows and columns. The kernel rechecks VL in the same streaming region as its
loads and stores; a mismatch rejects the operation before writing output.
Nothing assumes a particular M4 streaming vector length.

The C entry points have ordinary, non-streaming interfaces for cgo.
`__arm_locally_streaming` and `__arm_new("za")` let Clang manage streaming mode,
private ZA state, callee-saved registers and the AAPCS64 lazy-save protocol.
There is no handwritten Go/SME calling convention and no global `-march` flag.
See the [Arm ACLE specification](https://arm-software.github.io/acle/main/acle.html#sme-language-extensions-and-intrinsics).

## Running

Packing, scalar parity, dimension/guard checks and feature-gate tests run on any
ordinary Go build:

```sh
CGO_ENABLED=0 go test ./benchmarks/sme -count=1
```

Compiling the hardware experiment requires macOS ARM64, cgo, Apple Clang 21
(or another compatible SME compiler), and an explicit build tag. Execution
additionally requires `GOPHERLLM_SME_EXPERIMENT=1`, an empty
`GOPHERLLM_DISABLE_SIMD`, and successful nonzero read-only feature queries for
`hw.optional.arm.FEAT_SME` and `hw.optional.arm.SME_I8I32`. Missing or denied
queries fail closed. Even the instruction used to query VL is behind this gate.
These exact feature names come from Apple's
[XNU feature list](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/arm/arm_features.inc)
and [sysctl registration](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_mib.c).
iOS deliberately retains the stub in this experiment.

On an M4 or another compatible Mac, first run hardware parity:

```sh
CC="$(xcrun --find clang)" CGO_ENABLED=1 GOPHERLLM_SME_EXPERIMENT=1 \
  go test ./benchmarks/sme -tags=gopherllm_sme_experiment \
  -run '^TestSMETileBitExact$' -count=1 -v
```

Check that the test **passes**, rather than skips, and records the detected VL.
It checks full and partial tiles, depth tails, input/output guards, immutable
inputs, full-range int8 values and rejection of a changed VL. Then measure on
an otherwise idle machine:

```sh
CC="$(xcrun --find clang)" CGO_ENABLED=1 GOPHERLLM_SME_EXPERIMENT=1 GOMAXPROCS=1 \
  go test ./benchmarks/sme -tags=gopherllm_sme_experiment -run '^$' \
  -bench '^BenchmarkSMEInt8Tile$' -benchmem -benchtime=300ms -count=5
```

The three modes compare a scalar dot over packed buffers, SME over prepacked
buffers, and SME including packing both operands. Depths are 256, 3072 and
9216; tile rows are four and the detected lane count. Every benchmark checks
hardware output against the row-major scalar oracle before timing. The SME
measurements include cgo and streaming/ZA entry/exit overhead. These are tile
measurements, not model prefill speedups.

A machine without SME can still compile and audit the C object without running
optional instructions:

```sh
CLANG="$(xcrun --find clang)"
"$CLANG" -O3 -target arm64-apple-macos15.0 \
  -c benchmarks/sme/tile_darwin_arm64.c -o /private/tmp/gopherllm-sme-tile.o
"$(xcrun --find llvm-objdump)" -dr /private/tmp/gopherllm-sme-tile.o
```

The audit should show SMOPA, predicated byte loads and int32 ZA stores,
SMSTART/SMSTOP around locally-streaming functions, and the private-ZA save
protocol. Compilation/object inspection verifies code generation, not hardware
numerical parity or performance. M4 hardware validation remains required.
