#!/usr/bin/env python3
"""Export one dense Qwen/Ministral SwiGLU block for optional Core ML offload.

Install the optional offline dependencies with Python 3.12:
    python -m pip install 'coremltools>=9,<10' 'gguf>=0.19,<0.20'
    python scripts/export_coreml_ffn.py model.gguf layer0.mlpackage \
        --layer 0 --tokens 32 --compile-to layer0.mlmodelc

The fixed float32 interface is hidden[T,D] -> output[T,D]. Only the gate,
up and down projections and SiLU multiplication are exported. Normalization,
residuals, attention and KV/recurrent state remain with GopherLLM. Internally
the graph uses FP16 1x1 convolutions; optional palette8 storage also changes
weights numerically. Check parity and timing before enabling an offload.
Core ML chooses placement; CPU_AND_NE permission is not proof of ANE use.
--hidden-chunk splits wide intermediate channels into smaller convolutions
and sums their down projections; this changes FP16 reduction order.
SiLU uses a stable exp/div decomposition: Core ML's native Swish approximation
can introduce sizeable errors on the Neural Engine for real decoder gates.
--activation-scale scales gate/up weights by an exact power of two and keeps
their activations scaled through the down projection to avoid FP16 underflow.
The final output is divided by the square of this factor in FP32. Scaling
reduces FP16 headroom, so the runtime must validate each exported block.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile

from export_coreml_embeddings import Weights, compress_conv_weights, metadata, validate_weight_compression


ARCHITECTURES = frozenset(("qwen2", "qwen3", "qwen35", "ministral", "mistral3"))


@dataclass(frozen=True)
class FFNSpec:
    architecture: str
    layer: int
    dimension: int
    hidden_dimension: int
    tokens: int


def integer(value, label, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer within {minimum}..{maximum}")
    return value


def validate_arguments(layer, tokens):
    integer(layer, "layer", 0, 100000)
    integer(tokens, "tokens", 1, 128)


def validate_hidden_chunk(hidden_chunk):
    return integer(hidden_chunk, "hidden chunk", 0, 16384)


def validate_activation_scale(scale):
    integer(scale, "activation scale", 1, 4096)
    if scale & (scale - 1):
        raise ValueError("activation scale must be a power of two")
    return scale


def read_spec(reader, layer, tokens):
    validate_arguments(layer, tokens)
    architecture = metadata(reader, "general.architecture")
    if architecture not in ARCHITECTURES:
        raise ValueError("only dense Qwen and Ministral SwiGLU blocks are supported")
    count = integer(metadata(reader, architecture + ".block_count"), "block_count", 1, 100000)
    if layer >= count:
        raise ValueError(f"layer {layer} exceeds the {count} source blocks")
    dimension = integer(metadata(reader, architecture + ".embedding_length"), "embedding_length", 1, 16384)
    hidden = metadata(reader, architecture + ".feed_forward_length")
    if isinstance(hidden, list):
        if len(hidden) != count:
            raise ValueError("per-layer feed_forward_length must describe every source block")
        hidden = hidden[layer]
    hidden = integer(hidden, "feed_forward_length", 1, 262144)
    if tokens * dimension > 1 << 20:
        raise ValueError("tokens * embedding_length exceeds the 1048576-element I/O limit")
    for key in ("expert_count", "expert_used_count", "expert_shared_count"):
        if metadata(reader, architecture + "." + key, 0) != 0:
            raise ValueError("MoE projections require a different graph")
    spec = FFNSpec(architecture, layer, dimension, hidden, tokens)
    tensor_shapes(reader, spec)
    return spec


def tensor_shapes(reader, spec):
    """Validate the entire selected block before allocating dequantized weights."""
    prefix = f"blk.{spec.layer}."
    tensors = {}
    for tensor in reader.tensors:
        if tensor.name in tensors:
            raise ValueError(f"duplicate source tensor {tensor.name}")
        tensors[tensor.name] = tensor
    if prefix + "ffn_gate_up.weight" in tensors or prefix + "ffn_gate_inp.weight" in tensors:
        raise ValueError("fused gate/up and routed projections are not supported")
    shapes = ((prefix + "ffn_gate.weight", (spec.hidden_dimension, spec.dimension)),
              (prefix + "ffn_up.weight", (spec.hidden_dimension, spec.dimension)),
              (prefix + "ffn_down.weight", (spec.dimension, spec.hidden_dimension)))
    for name, shape in shapes:
        if name not in tensors:
            raise ValueError(f"missing tensor {name}")
        actual = tuple(int(value) for value in tensors[name].shape[::-1])
        if actual != shape:
            raise ValueError(f"{name}: shape {actual}, expected {shape}")
        if name.removesuffix(".weight") + ".bias" in tensors:
            raise ValueError(f"projection bias is not supported: {name}")
    return shapes, tensors


def source_fingerprints(reader, spec):
    shapes, tensors = tensor_shapes(reader, spec)
    result = []
    for name, shape in shapes:
        tensor = tensors[name]
        # GGUFReader exposes contiguous mmap-backed encoded bytes. Hash those
        # bytes directly: no full-source read or decoded FP32 weight copy.
        raw = memoryview(tensor.data).cast("B")
        result.append({"name": name, "shape": list(shape), "sha256": hashlib.sha256(raw).hexdigest(),
                       "tensor_type": int(tensor.tensor_type)})
    return result


def stable_silu(mb, np, x, scale=1):
    """Return scale*SiLU(x/scale) without Swish error or exp overflow.

    exp(-abs(x)/scale) lies in [0, 1]. Keep the numerator scaled until the
    down projection. Do not express this as x*sigmoid(x/scale): conversion
    can fuse that form back to the approximate Swish op when scale is one.
    """
    exponent = mb.exp(x=mb.mul(x=mb.abs(x=x), y=np.float16(-1 / scale)))
    numerator = mb.select(cond=mb.greater_equal(x=x, y=np.float16(0)),
                          a=x, b=mb.mul(x=x, y=exponent))
    return mb.real_div(x=numerator, y=mb.add(x=exponent, y=np.float16(1)))


def build_program(reader, spec, hidden_chunk=0, activation_scale=16):
    hidden_chunk = validate_hidden_chunk(hidden_chunk)
    activation_scale = validate_activation_scale(activation_scale)
    import numpy as np
    from gguf.quants import dequantize
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    tensor_shapes(reader, spec)
    weights = Weights(reader, np, dequantize)
    prefix = f"blk.{spec.layer}."
    gate = weights.get(prefix + "ffn_gate.weight", (spec.hidden_dimension, spec.dimension))[:, :, None, None]
    up = weights.get(prefix + "ffn_up.weight", (spec.hidden_dimension, spec.dimension))[:, :, None, None]
    down = weights.get(prefix + "ffn_down.weight", (spec.dimension, spec.hidden_dimension))[:, :, None, None]
    with np.errstate(over="ignore", invalid="ignore"):
        gate = gate * np.float16(activation_scale)
        up = up * np.float16(activation_scale)
    if not np.isfinite(gate).all() or not np.isfinite(up).all():
        raise ValueError("activation scale overflows FP16 projection weights; choose a smaller scale")

    @mb.program(input_specs=[mb.TensorSpec(shape=(spec.tokens, spec.dimension), dtype=types.fp32)],
                opset_version=ct.target.iOS16)
    def program(hidden):
        channels = mb.transpose(x=mb.cast(x=hidden, dtype="fp16"), perm=[1, 0])
        channels = mb.reshape(x=channels, shape=[1, spec.dimension, 1, spec.tokens])
        chunk = hidden_chunk or spec.hidden_dimension
        result = None
        for begin in range(0, spec.hidden_dimension, chunk):
            end = min(begin + chunk, spec.hidden_dimension)
            gated = mb.conv(x=channels, weight=gate[begin:end], name=f"gate_projection_{begin}")
            projected = mb.conv(x=channels, weight=up[begin:end], name=f"up_projection_{begin}")
            activated = mb.mul(x=stable_silu(mb, np, gated, scale=activation_scale), y=projected)
            partial = mb.conv(x=activated, weight=down[:, begin:end], name=f"down_projection_{begin}")
            result = partial if result is None else mb.add(x=result, y=partial)
        result = mb.reshape(x=result, shape=[spec.dimension, spec.tokens])
        result = mb.transpose(x=result, perm=[1, 0])
        result = mb.cast(x=result, dtype="fp32")
        return mb.mul(x=result, y=np.float32(1 / activation_scale**2), name="output")

    return program


def manifest(spec, fingerprints, weight_compression, model, hidden_chunk=0, activation_scale=16):
    return {"version": 1, "model": model, "architecture": spec.architecture, "layer": spec.layer,
            "dimension": spec.dimension, "hidden_dimension": spec.hidden_dimension, "tokens": spec.tokens,
            "input": "hidden", "output": "output", "compute_precision": "float16",
            "weight_compression": weight_compression, "hidden_chunk": hidden_chunk,
            "activation": "silu_exp", "activation_scale": activation_scale, "weights": fingerprints}


def export(source, output, layer=0, tokens=32, compile_to=None, weight_compression="none", hidden_chunk=0, activation_scale=16):
    validate_arguments(layer, tokens)
    hidden_chunk = validate_hidden_chunk(hidden_chunk)
    activation_scale = validate_activation_scale(activation_scale)
    weight_compression = validate_weight_compression(weight_compression)
    output = Path(output)
    if output.suffix != ".mlpackage":
        raise ValueError("output must have the .mlpackage suffix")
    destinations = [output, output.with_suffix(".json")]
    if compile_to is not None:
        compile_to = Path(compile_to)
        if compile_to.suffix != ".mlmodelc":
            raise ValueError("--compile-to must have the .mlmodelc suffix")
        destinations.extend((compile_to, compile_to.with_suffix(".json")))
    for path in destinations:
        if path.exists():
            raise ValueError(f"output already exists: {path}")

    import coremltools as ct
    import gguf

    reader = gguf.GGUFReader(str(source))
    spec = read_spec(reader, layer, tokens)
    fingerprints = source_fingerprints(reader, spec)
    program = build_program(reader, spec, hidden_chunk=hidden_chunk, activation_scale=activation_scale)
    # Projections and activations are already explicitly FP16. Disable the global
    # downcast pass so the final FP32 unscale stays FP32 on CPU/GPU; lowering
    # that last multiplication to FP16 would flush small final values on ANE.
    model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16,
                       compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_AND_NE,
                       skip_model_load=True)
    model = compress_conv_weights(model, weight_compression)
    model.short_description = "GopherLLM dense SwiGLU FFN; fixed FP32 I/O, FP16 compute, no attention or cache"
    contract = manifest(spec, fingerprints, weight_compression, output.name, hidden_chunk=hidden_chunk,
                        activation_scale=activation_scale)
    model.user_defined_metadata["gopherllm.ffn"] = json.dumps(contract, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Save and compile before publishing. Failed conversions or compilation
    # leave neither a partial model nor a misleading success manifest.
    with tempfile.TemporaryDirectory(prefix=".gopherllm-coreml-ffn-", dir=output.parent) as temporary:
        stage = Path(temporary)
        package = stage / output.name
        model.save(str(package))
        compiled = None
        if compile_to is not None:
            compile_to.parent.mkdir(parents=True, exist_ok=True)
            # Core ML's compiler expects an explicit .mlmodelc destination.
            compiled = stage / compile_to.name
            ct.models.utils.compile_model(str(package), destination_path=str(compiled))
        manifests = {output.with_suffix(".json"): contract}
        if compile_to is not None:
            manifests[compile_to.with_suffix(".json")] = dict(contract, model=compile_to.name)
        # Check again after potentially expensive conversion. The usual
        # single-writer CLI contract must never replace an existing export.
        for path in destinations:
            if path.exists():
                raise ValueError(f"output already exists: {path}")
        package.rename(output)
        if compiled is not None:
            compiled.rename(compile_to)
        for path, value in manifests.items():
            with path.open("x") as file:
                json.dump(value, file, sort_keys=True, indent=2)
                file.write("\n")
    return dict(contract, source=str(source), model=str(output),
                compiled_model=str(compile_to) if compile_to is not None else None,
                manifest=str((compile_to or output).with_suffix(".json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="matching dense Qwen or Ministral GGUF checkpoint")
    parser.add_argument("output", help="new .mlpackage destination")
    parser.add_argument("--layer", type=int, default=0, help="zero-based transformer block (default: 0)")
    parser.add_argument("--tokens", type=int, default=32, help="fixed independent token rows, 1..128 (default: 32)")
    parser.add_argument("--activation-scale", type=int, default=16,
                        help="power-of-two gate/up weight factor, FP32 output divides by its square (default: 16)")
    parser.add_argument("--hidden-chunk", type=int, default=0,
                        help="optional maximum intermediate channels per convolution, 1..16384; 0 keeps the full width")
    parser.add_argument("--weight-compression", choices=("none", "palette8"), default="none",
                        help="optional 256-entry scalar conv-weight palette (default: none)")
    parser.add_argument("--compile-to", help="optionally compile once to this new persistent .mlmodelc path")
    args = parser.parse_args()
    try:
        result = export(args.source, args.output, layer=args.layer, tokens=args.tokens,
                        compile_to=args.compile_to, weight_compression=args.weight_compression, hidden_chunk=args.hidden_chunk,
                        activation_scale=args.activation_scale)
    except (ValueError, ImportError, OSError) as error:
        parser.exit(1, f"export_coreml_ffn: {error}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
