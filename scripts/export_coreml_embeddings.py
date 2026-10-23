#!/usr/bin/env python3
"""Export a GGUF BERT encoder to a static, FP16 Core ML embedding graph.

This is an optional offline tool, not a runtime dependency. Install Apple's
coremltools and the ggml-org GGUF reader in a disposable Python environment:

    python -m pip install 'coremltools>=9,<10' gguf numpy
    python scripts/export_coreml_embeddings.py encoder.gguf encoder.mlpackage \
        --sequence-length 128 --batch-size 4 --compile-to encoder.mlmodelc

Use the exact source GGUF's tokenizer with the exported model. The graph emits
the source pooling mode; GopherLLM's Core ML adapter performs L2 normalization.
It enables Neural Engine execution but does not guarantee device placement.
Static batches share one weight bank; unused trailing rows may be fully masked.
For CLS pooling, --prune-cls computes only the retained query and feed-forward
position in the last block. Keys and values still cover the complete sequence.
--weight-compression palette8 applies an optional 256-entry palette only to
convolution weights. Compute remains FP16; vocabulary, biases and norms retain
their original exported values. Compare embedding quality before adopting it.
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class BERTSpec:
    dimension: int
    hidden: int
    heads: int
    layers: int
    context: int
    pooling: int
    epsilon: float
    sequence: int


def metadata(reader, name, default=None):
    field = reader.fields.get(name)
    return default if field is None else field.contents()


def read_spec(reader, sequence):
    if metadata(reader, "general.architecture") != "bert":
        raise ValueError("only GGUF bert encoders are supported; Nomic/RoPE and chat decoders require different graphs")
    if metadata(reader, "bert.attention.causal", False):
        raise ValueError("causal BERT attention is not supported")
    values = {}
    for key, field in (("dimension", "embedding_length"), ("hidden", "feed_forward_length"),
                       ("heads", "attention.head_count"), ("layers", "block_count"),
                       ("context", "context_length")):
        value = metadata(reader, "bert." + field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"bert.{field} must be a positive integer")
        values[key] = value
    if sequence < 1 or sequence > values["context"]:
        raise ValueError(f"sequence length must be within 1..{values['context']}")
    if values["dimension"] % values["heads"]:
        raise ValueError("embedding dimension must be divisible by attention heads")
    head_dim = values["dimension"] // values["heads"]
    for name in ("key_length", "value_length"):
        if metadata(reader, "bert.attention." + name, head_dim) != head_dim:
            raise ValueError("unequal BERT query/key/value dimensions are not supported")
    pooling = metadata(reader, "bert.pooling_type", 1)
    if pooling not in (1, 2):
        raise ValueError(f"unsupported BERT pooling type {pooling}; expected mean=1 or CLS=2")
    epsilon = metadata(reader, "bert.attention.layer_norm_epsilon", 1e-5)
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("layer norm epsilon must be positive and finite")
    return BERTSpec(**values, pooling=pooling, epsilon=epsilon, sequence=sequence)


class Weights:
    """Decode one tensor at a time, with bounded FP32 dequantization scratch."""

    def __init__(self, reader, np, dequantize):
        self.tensors = {tensor.name: tensor for tensor in reader.tensors}
        self.np, self.dequantize = np, dequantize

    def shape(self, name):
        if name not in self.tensors:
            raise ValueError(f"missing tensor {name}")
        return tuple(int(n) for n in self.tensors[name].shape[::-1])

    def get(self, name, shape, optional=False, first_rows=None):
        np = self.np
        if optional and name not in self.tensors:
            return np.zeros(shape, dtype=np.float16)
        actual = self.shape(name)
        if actual != tuple(shape):
            raise ValueError(f"{name}: shape {actual}, expected {tuple(shape)}")
        tensor = self.tensors[name]
        if first_rows is not None:
            shape = (first_rows, *shape[1:])
        result = np.empty(shape, dtype=np.float16)
        if len(shape) == 1:
            result[:] = self.dequantize(tensor.data, tensor.tensor_type).reshape(-1)
            if not np.isfinite(result).all():
                raise ValueError(f"{name}: non-finite values or FP16 overflow")
        else:
            for begin in range(0, shape[0], 128):
                end = min(begin + 128, shape[0])
                result[begin:end] = self.dequantize(tensor.data[begin:end], tensor.tensor_type)
                if not np.isfinite(result[begin:end]).all():
                    raise ValueError(f"{name}: non-finite values or FP16 overflow")
        return result


def validate_batch_size(batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size not in (1, 2, 4, 8):
        raise ValueError("batch size must be one of 1, 2, 4, or 8")
    return batch_size


def validate_weight_compression(weight_compression):
    if not isinstance(weight_compression, str) or weight_compression not in ("none", "palette8"):
        raise ValueError("weight compression must be none or palette8")
    return weight_compression


def conv_weight_names(spec):
    """Select exclusively conv.weight constants using the public Model proto.

    Names are small graph metadata. Do not inspect tensor values or calculate
    statistics for the much larger vocabulary bank. This exporter produces a
    flat, single-function graph; reject other scopes rather than conflate names.
    """
    if spec.WhichOneof("Type") != "mlProgram":
        raise ValueError("weight compression requires an mlprogram model")
    functions = list(spec.mlProgram.functions.values())
    if len(functions) != 1:
        raise ValueError("weight compression requires one function")
    blocks = list(functions[0].block_specializations.values())
    if len(blocks) != 1 or any(op.blocks for block in blocks for op in block.operations):
        raise ValueError("weight compression requires one flat block")
    block = blocks[0]
    uses = {}
    for op in block.operations:
        for parameter, argument in op.inputs.items():
            for binding in argument.arguments:
                if binding.WhichOneof("binding") == "name":
                    uses.setdefault(binding.name, []).append((op.type, parameter))
    eligible_names = {}
    for op in block.operations:
        if op.type != "const" or len(op.outputs) != 1:
            continue
        output = op.outputs[0].name
        consumers = uses.get(output, ())
        eligible = bool(consumers) and output not in block.outputs and all(
            consumer == ("conv", "weight") for consumer in consumers)
        attribute = op.attributes.get("name")
        names = () if attribute is None else attribute.immediateValue.tensor.strings.values
        if len(names) != 1 or not names[0]:
            if eligible:
                raise ValueError("convolution weight constant lacks an unambiguous operation name")
            continue
        # Configuration keys are operation names, not output variable names.
        # A collision must not turn a non-convolution constant into a target.
        eligible_names[names[0]] = eligible_names.get(names[0], True) and eligible
    return sorted(name for name, eligible in eligible_names.items() if eligible)


def compress_conv_weights(model, weight_compression, weight_threshold=4096):
    """Optional scalar palette storage; operations continue to compute in FP16."""
    weight_compression = validate_weight_compression(weight_compression)
    if weight_compression == "none":
        return model
    if isinstance(weight_threshold, bool) or not isinstance(weight_threshold, int) or weight_threshold < 0:
        raise ValueError("weight threshold must be a nonnegative integer")
    names = conv_weight_names(model.get_spec())
    if not names:
        raise ValueError("no exclusively used convolution weight constants to compress")
    from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights

    palette = OpPalettizerConfig(mode="kmeans", nbits=8, granularity="per_tensor", cluster_dim=1,
                                num_kmeans_workers=1, enable_per_channel_scale=False,
                                weight_threshold=weight_threshold)
    config = OptimizationConfig(global_config=None, op_name_configs={name: palette for name in names})
    return palettize_weights(model, config=config, joint_compression=False)


def build_program(reader, spec, batch_size=1, prune_cls=False):
    batch_size = validate_batch_size(batch_size)
    if not isinstance(prune_cls, bool):
        raise ValueError("prune_cls must be a boolean")
    import numpy as np
    from gguf.quants import dequantize
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    weights = Weights(reader, np, dequantize)
    dim, hidden, sequence = spec.dimension, spec.hidden, spec.sequence
    vocab_shape = weights.shape("token_embd.weight")
    if len(vocab_shape) != 2 or vocab_shape[0] < 1 or vocab_shape[1] != dim:
        raise ValueError("token embedding must be [vocabulary, dimension]")
    position_shape = weights.shape("position_embd.weight")
    if len(position_shape) != 2 or position_shape[0] < sequence or position_shape[1] != dim:
        raise ValueError("position embedding does not cover the requested sequence")
    for name in weights.tensors:
        if name.endswith("ffn_gate.weight"):
            raise ValueError("gated BERT feed-forward layers are not supported")
    # Fail before decoding the potentially large vocabulary bank if this GGUF
    # uses an unsupported tensor layout or lacks a transformer layer.
    def require_shape(name, shape, optional=False):
        if optional and name not in weights.tensors:
            return
        if weights.shape(name) != tuple(shape):
            raise ValueError(f"{name}: shape {weights.shape(name)}, expected {tuple(shape)}")

    def require_norm(prefix):
        require_shape(prefix + ".weight", (dim,))
        require_shape(prefix + ".bias", (dim,), optional=True)

    def require_projection(prefix, rows, cols):
        require_shape(prefix + ".weight", (rows, cols))
        require_shape(prefix + ".bias", (rows,), optional=True)

    require_norm("token_embd_norm")
    for layer in range(spec.layers):
        prefix = f"blk.{layer}."
        if prefix + "attn_qkv.weight" in weights.tensors:
            require_projection(prefix + "attn_qkv", 3 * dim, dim)
        else:
            for name in ("attn_q", "attn_k", "attn_v"):
                require_projection(prefix + name, dim, dim)
        require_projection(prefix + "attn_output", dim, dim)
        require_projection(prefix + "ffn_up", hidden, dim)
        require_projection(prefix + "ffn_down", dim, hidden)
        require_norm(prefix + "attn_output_norm")
        require_norm(prefix + "layer_output_norm")
    eps = np.float16(spec.epsilon)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("layer norm epsilon is not representable in FP16")

    def norm(x, prefix):
        return mb.layer_norm(x=x, axes=[1],
                             gamma=weights.get(prefix + ".weight", (dim,)),
                             beta=weights.get(prefix + ".bias", (dim,), optional=True),
                             epsilon=eps)

    def project(x, prefix, rows, cols):
        return mb.conv(x=x,
                       weight=weights.get(prefix + ".weight", (rows, cols)).reshape(rows, cols, 1, 1),
                       bias=weights.get(prefix + ".bias", (rows,), optional=True))

    def cls_position(x):
        return mb.slice_by_index(x=x, begin=[0, 0, 0, 0], end=[batch_size, dim, 1, 1])

    def project_qkv(x, prefix, cls_query=None):
        if prefix + "attn_qkv.weight" in weights.tensors:
            qkv = project(x, prefix + "attn_qkv", 3 * dim, dim)
        else:
            return (project(x if cls_query is None else cls_query, prefix + "attn_q", dim, dim),
                    project(x, prefix + "attn_k", dim, dim), project(x, prefix + "attn_v", dim, dim))
        q, k, v = mb.split(x=qkv, num_splits=3, axis=1)
        return (q if cls_query is None else cls_position(q), k, v)

    @mb.program(input_specs=[mb.TensorSpec(shape=(batch_size, sequence), dtype=types.int32),
                             mb.TensorSpec(shape=(batch_size, sequence), dtype=types.int32)],
                opset_version=ct.target.iOS16)
    def program(input_ids, attention_mask):
        tokens = weights.get("token_embd.weight", vocab_shape)
        x = mb.gather(x=tokens, indices=input_ids, axis=0)
        x = mb.expand_dims(x=mb.transpose(x=x, perm=[0, 2, 1]), axes=[2])
        positions = weights.get("position_embd.weight", position_shape, first_rows=sequence)
        x = mb.add(x=x, y=positions.T.reshape(1, dim, 1, sequence))
        if "token_types.weight" in weights.tensors:
            type_shape = weights.shape("token_types.weight")
            if math.prod(type_shape) < dim:
                raise ValueError("token type embedding is shorter than the embedding dimension")
            type_embedding = weights.get("token_types.weight", type_shape).reshape(-1)[:dim]
            x = mb.add(x=x, y=type_embedding.reshape(1, dim, 1, 1))
        x = norm(x, "token_embd_norm")
        mask = mb.cast(x=attention_mask, dtype="fp16")
        # Scores are [batch, key, 1, query], so masking the channel axis
        # excludes padded keys without transposing the attention matrix.
        key_mask = mb.reshape(x=mask, shape=[batch_size, sequence, 1, 1])
        key_bias = mb.mul(x=mb.sub(x=np.float16(1), y=key_mask), y=np.float16(-10000))
        for layer in range(spec.layers):
            prefix = f"blk.{layer}."
            # Subsequent BERT operations are pointwise once attention has
            # mixed the values. Only the final CLS query can reach CLS pooling;
            # K/V must still be projected from every preceding-layer position.
            cls_only = prune_cls and spec.pooling == 2 and layer == spec.layers - 1
            residual = cls_position(x) if cls_only else x
            q, k, v = project_qkv(x, prefix, residual if cls_only else None)
            q_heads = mb.split(x=q, num_splits=spec.heads, axis=1) if spec.heads > 1 else [q]
            k_heads = mb.split(x=k, num_splits=spec.heads, axis=1) if spec.heads > 1 else [k]
            v_heads = mb.split(x=v, num_splits=spec.heads, axis=1) if spec.heads > 1 else [v]
            attended = []
            for query, key, value in zip(q_heads, k_heads, v_heads):
                key = mb.transpose(x=key, perm=[0, 3, 2, 1])
                scores = mb.einsum(values=(key, query), equation="bkhc,bchq->bkhq")
                scores = mb.mul(x=scores, y=np.float16(1 / math.sqrt(dim // spec.heads)))
                probabilities = mb.softmax(x=mb.add(x=scores, y=key_bias), axis=1)
                attended.append(mb.einsum(values=(value, probabilities), equation="bchk,bkhq->bchq"))
            attention = mb.concat(values=attended, axis=1)
            x = norm(mb.add(x=residual, y=project(attention, prefix + "attn_output", dim, dim)),
                     prefix + "attn_output_norm")
            up = project(x, prefix + "ffn_up", hidden, dim)
            up = mb.gelu(x=up, mode="EXACT")
            x = norm(mb.add(x=x, y=project(up, prefix + "ffn_down", dim, hidden)),
                     prefix + "layer_output_norm")
        if spec.pooling == 2:
            pooled = cls_position(x)
        else:
            pooling_mask = mb.reshape(x=mask, shape=[batch_size, 1, 1, sequence])
            total = mb.reduce_sum(x=mb.mul(x=x, y=pooling_mask), axes=[3], keep_dims=True)
            count = mb.reduce_sum(x=pooling_mask, axes=[3], keep_dims=True)
            pooled = mb.real_div(x=total, y=mb.maximum(x=count, y=np.float16(1)))
        pooled = mb.reshape(x=pooled, shape=[batch_size, dim])
        return mb.cast(x=pooled, dtype="fp32", name="embedding")

    return program


def export(source, output, sequence, compile_to=None, batch_size=1, prune_cls=False, weight_compression="none"):
    batch_size = validate_batch_size(batch_size)
    weight_compression = validate_weight_compression(weight_compression)
    if not isinstance(prune_cls, bool):
        raise ValueError("prune_cls must be a boolean")
    import coremltools as ct
    import gguf

    source, output = Path(source), Path(output)
    if output.suffix != ".mlpackage":
        raise ValueError("output must have the .mlpackage suffix")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if compile_to is not None:
        compile_to = Path(compile_to)
        if compile_to.suffix != ".mlmodelc" or compile_to.exists():
            raise ValueError("--compile-to must be a new .mlmodelc path")
    reader = gguf.GGUFReader(str(source))
    spec = read_spec(reader, sequence)
    program = build_program(reader, spec, batch_size=batch_size, prune_cls=prune_cls)
    model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16,
                       compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE,
                       skip_model_load=True)
    model = compress_conv_weights(model, weight_compression)
    compression_scope = "conv_weights" if weight_compression != "none" else "none"
    model.short_description = "GopherLLM static BERT encoder; FP16 compute, pooled output, external L2 normalization"
    model.user_defined_metadata["gopherllm.architecture"] = "bert"
    model.user_defined_metadata["gopherllm.sequence_length"] = str(sequence)
    model.user_defined_metadata["gopherllm.batch_size"] = str(batch_size)
    model.user_defined_metadata["gopherllm.prune_cls"] = str(prune_cls and spec.pooling == 2).lower()
    model.user_defined_metadata["gopherllm.weight_compression"] = weight_compression
    model.user_defined_metadata["gopherllm.weight_compression_scope"] = compression_scope
    model.user_defined_metadata["gopherllm.compute_precision"] = "float16"
    model.user_defined_metadata["gopherllm.pooling_type"] = str(spec.pooling)
    token_embedding = next(tensor for tensor in reader.tensors if tensor.name == "token_embd.weight")
    model.user_defined_metadata["gopherllm.vocabulary_size"] = str(int(token_embedding.shape[1]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gopherllm-coreml-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        model.save(str(staged))
        staged.rename(output)
    if compile_to is not None:
        compile_to.parent.mkdir(parents=True, exist_ok=True)
        ct.models.utils.compile_model(str(output), destination_path=str(compile_to))
    return {"source": str(source), "model": str(output), "compiled_model": str(compile_to) if compile_to else None,
            "sequence_length": sequence, "batch_size": batch_size,
            "prune_cls": prune_cls and spec.pooling == 2,
            "weight_compression": weight_compression, "weight_compression_scope": compression_scope,
            "dimension": spec.dimension, "layers": spec.layers,
            "pooling_type": spec.pooling, "precision": "float16", "compute_precision": "float16"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="matching GGUF bert embedding checkpoint")
    parser.add_argument("output", help="new .mlpackage destination")
    parser.add_argument("--sequence-length", type=int, default=128, help="static sequence length (default: 128)")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8), default=1,
                        help="static number of independent texts per prediction (default: 1)")
    parser.add_argument("--prune-cls", action="store_true",
                        help="compute only the retained query/output/FFN position in the last CLS-pooling block")
    parser.add_argument("--weight-compression", choices=("none", "palette8"), default="none",
                        help="optional conv-weight palette; vocabulary/bias/norm values remain FP16 (default: none)")
    parser.add_argument("--compile-to", help="optionally compile once to this new persistent .mlmodelc path (macOS)")
    args = parser.parse_args()
    try:
        result = export(args.source, args.output, args.sequence_length, args.compile_to,
                        batch_size=args.batch_size, prune_cls=args.prune_cls, weight_compression=args.weight_compression)
    except (ValueError, ImportError, OSError) as error:
        parser.exit(1, f"export_coreml_embeddings: {error}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
