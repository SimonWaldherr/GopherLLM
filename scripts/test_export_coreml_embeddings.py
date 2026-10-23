"""Header/argument validation needs no Core ML runtime, weights or Apple hardware."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import math
import os
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_coreml_embeddings as exporter
from export_coreml_embeddings import build_program, conv_weight_names, read_spec, validate_batch_size


class Field:
    def __init__(self, value):
        self.value = value

    def contents(self):
        return self.value


def fixture(**overrides):
    metadata = {
        "general.architecture": "bert", "bert.embedding_length": 768,
        "bert.feed_forward_length": 3072, "bert.attention.head_count": 12,
        "bert.block_count": 12, "bert.context_length": 512,
        "bert.pooling_type": 2, "bert.attention.layer_norm_epsilon": 1e-5,
    }
    metadata.update(overrides)
    return types.SimpleNamespace(fields={key: Field(value) for key, value in metadata.items()})


class HeaderValidation(unittest.TestCase):
    def test_granite_pooling_and_shape(self):
        spec = read_spec(fixture(), 128)
        self.assertEqual((spec.dimension, spec.heads, spec.pooling, spec.sequence), (768, 12, 2, 128))

    def test_rejects_graphs_requiring_other_semantics(self):
        for changes in ({"general.architecture": "nomic-bert"},
                        {"general.architecture": "mistral3"},
                        {"bert.attention.causal": True},
                        {"bert.pooling_type": 3},
                        {"bert.attention.key_length": 32}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                read_spec(fixture(**changes), 128)

    def test_rejects_invalid_static_geometry(self):
        for sequence in (0, -1, 513):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                read_spec(fixture(), sequence)
        for changes in ({"bert.attention.head_count": 0},
                        {"bert.attention.head_count": 7},
                        {"bert.block_count": True},
                        {"bert.embedding_length": -1},
                        {"bert.attention.layer_norm_epsilon": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                read_spec(fixture(**changes), 128)


class BatchArguments(unittest.TestCase):
    def test_accepts_only_supported_integer_batch_sizes(self):
        for batch_size in (1, 2, 4, 8):
            self.assertEqual(validate_batch_size(batch_size), batch_size)
        for batch_size in (None, True, False, 0, -1, 3, 16, 2.0, "2"):
            with self.subTest(batch_size=batch_size):
                # Both public entry points validate before optional imports or
                # model access, so malformed sizes fail on any host.
                with self.assertRaisesRegex(ValueError, "batch size"):
                    build_program(None, None, batch_size=batch_size)
                with self.assertRaisesRegex(ValueError, "batch size"):
                    exporter.export(None, None, 8, batch_size=batch_size)

    def test_cli_forwards_default_and_explicit_batches(self):
        for batch_size in (None, 1, 2, 4, 8):
            argv = ["export_coreml_embeddings.py", "encoder.gguf", "encoder.mlpackage"]
            if batch_size is not None:
                argv.extend(["--batch-size", str(batch_size)])
            with self.subTest(batch_size=batch_size), patch.object(sys, "argv", argv), \
                    patch.object(exporter, "export", return_value={}) as export_mock, \
                    redirect_stdout(StringIO()):
                exporter.main()
                export_mock.assert_called_once_with("encoder.gguf", "encoder.mlpackage", 128, None,
                                                    batch_size=1 if batch_size is None else batch_size,
                                                    prune_cls=False, weight_compression="none")

    def test_cli_forwards_cls_pruning(self):
        argv = ["export_coreml_embeddings.py", "encoder.gguf", "encoder.mlpackage", "--prune-cls"]
        with patch.object(sys, "argv", argv), patch.object(exporter, "export", return_value={}) as export_mock, \
                redirect_stdout(StringIO()):
            exporter.main()
        export_mock.assert_called_once_with("encoder.gguf", "encoder.mlpackage", 128, None,
                                            batch_size=1, prune_cls=True, weight_compression="none")

    def test_weight_compression_is_optional_and_validated_before_imports(self):
        marker = object()
        self.assertIs(exporter.compress_conv_weights(marker, "none"), marker)
        for mode in (None, True, False, 8, "linear8", "palette4", "PALETTE8"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "weight compression"):
                exporter.export(None, None, 8, weight_compression=mode)
        for mode in ("none", "palette8"):
            self.assertEqual(exporter.validate_weight_compression(mode), mode)

    def test_cli_forwards_weight_compression(self):
        argv = ["export_coreml_embeddings.py", "encoder.gguf", "encoder.mlpackage", "--weight-compression", "palette8"]
        with patch.object(sys, "argv", argv), patch.object(exporter, "export", return_value={}) as export_mock, \
                redirect_stdout(StringIO()):
            exporter.main()
        export_mock.assert_called_once_with("encoder.gguf", "encoder.mlpackage", 128, None,
                                            batch_size=1, prune_cls=False, weight_compression="palette8")

    def test_cls_pruning_requires_boolean_before_imports(self):
        for prune_cls in (None, 0, 1, "false", "true"):
            with self.subTest(prune_cls=prune_cls):
                with self.assertRaisesRegex(ValueError, "prune_cls"):
                    build_program(None, None, prune_cls=prune_cls)
                with self.assertRaisesRegex(ValueError, "prune_cls"):
                    exporter.export(None, None, 8, prune_cls=prune_cls)

    def test_cli_rejects_unsupported_batch_before_export(self):
        for batch_size in ("0", "3", "16", "2.0"):
            argv = ["export_coreml_embeddings.py", "encoder.gguf", "encoder.mlpackage",
                    "--batch-size", batch_size]
            with self.subTest(batch_size=batch_size), patch.object(sys, "argv", argv), \
                    patch.object(exporter, "export") as export_mock, redirect_stderr(StringIO()), \
                    self.assertRaises(SystemExit) as error:
                exporter.main()
            self.assertEqual(error.exception.code, 2)
            export_mock.assert_not_called()


class CompressionSelection(unittest.TestCase):
    @staticmethod
    def operation(kind, output, operation_name=None, **inputs):
        attributes = {}
        if operation_name is not None:
            attributes["name"] = types.SimpleNamespace(immediateValue=types.SimpleNamespace(
                tensor=types.SimpleNamespace(strings=types.SimpleNamespace(values=[operation_name]))))
        return types.SimpleNamespace(type=kind, outputs=[types.SimpleNamespace(name=output)], blocks=[],
                                     inputs={parameter: types.SimpleNamespace(arguments=[types.SimpleNamespace(
                                         name=value, WhichOneof=lambda _: "name")]) for parameter, value in inputs.items()},
                                     attributes=attributes)

    @staticmethod
    def spec(operations, outputs=()):
        block = types.SimpleNamespace(operations=operations, outputs=outputs)
        function = types.SimpleNamespace(block_specializations={"CoreML6": block})
        return types.SimpleNamespace(WhichOneof=lambda _: "mlProgram",
                                     mlProgram=types.SimpleNamespace(functions={"main": function}))

    def test_selects_constant_operation_names_exclusively_used_as_conv_weights(self):
        op = self.operation
        operations = [op("const", "kernel_variable", "kernel_operation"), op("const", "bias", "bias_operation"),
                      op("const", "vocabulary", "vocabulary_operation"), op("const", "norm", "norm_operation"),
                      op("conv", "projection", "conv_operation", weight="kernel_variable", bias="bias", x="input"),
                      op("conv", "second_projection", "another_conv", weight="kernel_variable", bias="bias", x="input"),
                      op("gather", "tokens", "lookup", x="vocabulary", indices="ids"),
                      op("layer_norm", "normalized", "normalize", x="projection", gamma="norm")]
        self.assertEqual(conv_weight_names(self.spec(operations)), ["kernel_operation"])

    def test_rejects_shared_nonweight_uses_outputs_and_ambiguous_names(self):
        op = self.operation
        for consumer in (op("conv", "projection", "projection", weight="shared", bias="shared"),
                         op("gather", "lookup", "lookup", x="shared", indices="ids")):
            operations = [op("const", "shared", "kernel"),
                          op("conv", "valid_use", "valid_use", weight="shared", x="input"), consumer]
            self.assertEqual(conv_weight_names(self.spec(operations)), [])
        operations = [op("const", "kernel_variable", "kernel"),
                      op("conv", "projection", "projection", weight="kernel_variable", x="input")]
        self.assertEqual(conv_weight_names(self.spec(operations, outputs=["kernel_variable"])), [])
        operations[0].attributes = {}
        with self.assertRaisesRegex(ValueError, "operation name"):
            conv_weight_names(self.spec(operations))
        # A name collision cannot make another constant's bias compressible.
        operations[0] = op("const", "kernel_variable", "same_name")
        operations.extend([op("const", "bias", "same_name"),
                           op("conv", "other_projection", "other", weight="kernel_variable", bias="bias", x="input")])
        self.assertEqual(conv_weight_names(self.spec(operations)), [])

    def test_rejects_scopes_outside_the_exporter_contract(self):
        op = self.operation
        spec = self.spec([])
        spec.WhichOneof = lambda _: "neuralNetwork"
        with self.assertRaisesRegex(ValueError, "mlprogram"):
            conv_weight_names(spec)
        spec = self.spec([])
        spec.mlProgram.functions["other"] = spec.mlProgram.functions["main"]
        with self.assertRaisesRegex(ValueError, "one function"):
            conv_weight_names(spec)
        nested = op("cond", "result", "branch")
        nested.blocks = [types.SimpleNamespace(operations=[], outputs=[])]
        with self.assertRaisesRegex(ValueError, "flat block"):
            conv_weight_names(self.spec([nested]))


@unittest.skipUnless(os.environ.get("GOPHERLLM_COREML_EXPORT_TESTS") == "1" and sys.platform == "darwin",
                     "set GOPHERLLM_COREML_EXPORT_TESTS=1 on macOS for native graph parity")
class GraphParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
            import coremltools as ct
            import gguf
        except ImportError as error:
            raise unittest.SkipTest(f"optional export dependencies unavailable: {error}")
        cls.np, cls.ct, cls.gguf = np, ct, gguf

    def tiny_reader(self, layers=1):
        np = self.np
        rng = np.random.default_rng(71253)
        weights = {}

        def add(name, shape, scale=.15, offset=0):
            weights[name] = (rng.normal(size=shape) * scale + offset).astype(np.float32)

        add("token_embd.weight", (16, 8), .3)
        add("position_embd.weight", (8, 8), .2)
        add("token_types.weight", (2, 8), .1)
        norms = ["token_embd_norm"]
        for layer in range(layers):
            norms.extend((f"blk.{layer}.attn_output_norm", f"blk.{layer}.layer_output_norm"))
        for name in norms:
            add(name + ".weight", (8,), .1, 1)
            add(name + ".bias", (8,), .05)
        for layer in range(layers):
            for name, rows, cols in (("attn_q", 8, 8), ("attn_k", 8, 8), ("attn_v", 8, 8),
                                     ("attn_output", 8, 8), ("ffn_up", 16, 8), ("ffn_down", 8, 16)):
                add(f"blk.{layer}." + name + ".weight", (rows, cols))
                add(f"blk.{layer}." + name + ".bias", (rows,), .05)
        reader = fixture(**{"bert.embedding_length": 8, "bert.feed_forward_length": 16,
                            "bert.attention.head_count": 2, "bert.block_count": layers,
                            "bert.context_length": 8})
        reader.tensors = [types.SimpleNamespace(name=name, shape=value.shape[::-1], data=value,
                                                tensor_type=self.gguf.GGMLQuantizationType.F32)
                          for name, value in weights.items()]
        # The exporter stores FP16 weights; evaluate those same values using
        # ordinary dense [sequence, channels] NumPy operations in float64.
        reference = {name: value.astype(np.float16).astype(np.float64) for name, value in weights.items()}
        return reader, reference

    def reference(self, weights, spec, ids, mask):
        np = self.np

        def norm(x, prefix):
            centered = x - x.mean(axis=-1, keepdims=True)
            return (centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + spec.epsilon)
                    * weights[prefix + ".weight"] + weights[prefix + ".bias"])

        def project(x, prefix):
            return x @ weights[prefix + ".weight"].T + weights[prefix + ".bias"]

        x = weights["token_embd.weight"][ids] + weights["position_embd.weight"][:len(ids)]
        x = norm(x + weights["token_types.weight"][0], "token_embd_norm")
        width = spec.dimension // spec.heads
        for layer in range(spec.layers):
            prefix = f"blk.{layer}."
            query, key, value = (project(x, prefix + name) for name in ("attn_q", "attn_k", "attn_v"))
            attended = []
            for head in range(spec.heads):
                channels = slice(head * width, (head + 1) * width)
                scores = query[:, channels] @ key[:, channels].T / math.sqrt(width)
                scores[:, mask == 0] -= 10000
                probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
                probabilities /= probabilities.sum(axis=-1, keepdims=True)
                attended.append(probabilities @ value[:, channels])
            x = norm(x + project(np.concatenate(attended, axis=-1), prefix + "attn_output"), prefix + "attn_output_norm")
            up = project(x, prefix + "ffn_up")
            activated = .5 * up * (1 + np.vectorize(math.erf)(up / math.sqrt(2)))
            x = norm(x + project(activated, prefix + "ffn_down"), prefix + "layer_output_norm")
        pooled = x[0] if spec.pooling == 2 else x[mask != 0].mean(axis=0)
        return pooled / np.linalg.norm(pooled)

    def test_batched_mean_and_cls_match_independent_dense_references(self):
        np, ct = self.np, self.ct
        lengths = {1: ((3,), (6,), (0,)), 2: ((3, 6), (6, 0)),
                   4: ((3, 6, 2, 0), (6, 2, 5, 0)),
                   8: ((3, 6, 2, 5, 4, 0, 0, 0), (6, 2, 5, 3, 4, 7, 1, 0))}
        variants = [(batch, False) for batch in (1, 2, 4, 8)]
        variants.extend((batch, True) for batch in (1, 4))
        for pooling in (1, 2):
            for batch_size, prune_cls in variants:
                # Two blocks expose accidental pruning of an earlier block.
                reader, weights = self.tiny_reader(layers=2 if prune_cls else 1)
                reader.fields["bert.pooling_type"] = Field(pooling)
                spec = read_spec(reader, 8)
                program = build_program(reader, spec, batch_size=batch_size, prune_cls=prune_cls)
                function = program.functions["main"]
                self.assertEqual(tuple(function.inputs["input_ids"].shape), (batch_size, 8))
                self.assertEqual(tuple(function.outputs[0].shape), (batch_size, 8))
                token_bank = weights["token_embd.weight"].astype(np.float16)
                bank_count = sum(op.op_type == "const" and np.array_equal(op.outputs[0].val, token_bank)
                                 for op in function.operations)
                self.assertEqual(bank_count, 1, "static batching duplicated the token weight bank")
                projections = [op for op in function.operations if op.op_type == "conv"]
                block_projections = 6
                self.assertEqual(len(projections), spec.layers * block_projections)
                retained_width = 1 if prune_cls and pooling == 2 else 8
                self.assertEqual(projections[-3].outputs[0].shape[-1], retained_width)
                self.assertEqual(projections[-1].outputs[0].shape[-1], retained_width)
                for earlier in projections[:-block_projections]:
                    self.assertEqual(earlier.outputs[0].shape[-1], 8, "an earlier block was pruned")
                softmaxes = [op for op in function.operations if op.op_type == "softmax"]
                self.assertEqual(tuple(softmaxes[-1].outputs[0].shape), (batch_size, 8, 1, retained_width))
                # K/V projections must still cover every sequence position.
                final_input_projections = projections[-block_projections:-3]
                self.assertEqual(final_input_projections[-1].outputs[0].shape[-1], 8)
                self.assertEqual(final_input_projections[0].outputs[0].shape[-1], retained_width)
                self.assertEqual(final_input_projections[1].outputs[0].shape[-1], 8)
                model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16,
                                   compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_ONLY)
                for valid_lengths in lengths[batch_size]:
                    outputs = []
                    mask = (np.arange(8)[None, :] < np.array(valid_lengths)[:, None]).astype(np.int32)
                    for padding in (0, 15):
                        with self.subTest(pooling=pooling, batch_size=batch_size, prune_cls=prune_cls,
                                          valid_lengths=valid_lengths, padding=padding):
                            ids = np.full((batch_size, 8), padding, dtype=np.int32)
                            for row, valid in enumerate(valid_lengths):
                                ids[row, :valid] = (np.arange(valid) + 1 + 3 * row) % 16
                            output = np.asarray(model.predict({"input_ids": ids, "attention_mask": mask})["embedding"],
                                                dtype=np.float64)
                            self.assertEqual(output.shape, (batch_size, 8))
                            self.assertTrue(np.isfinite(output).all())
                            real_outputs = []
                            for row, valid in enumerate(valid_lengths):
                                if not valid:
                                    if pooling == 1:
                                        np.testing.assert_array_equal(output[row], np.zeros(8))
                                    continue
                                norm = np.linalg.norm(output[row])
                                self.assertGreater(norm, 0)
                                normalized = output[row] / norm
                                # Evaluate each real text separately, without
                                # padded tokens or another batch row present.
                                reference = self.reference(weights, spec, ids[row, :valid],
                                                           np.ones(valid, dtype=np.int32))
                                np.testing.assert_allclose(normalized, reference, atol=3e-3, rtol=3e-3)
                                real_outputs.append(normalized)
                            outputs.append(np.asarray(real_outputs))
                    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-5, rtol=1e-5,
                                               err_msg="padded token IDs changed a real embedding")
                    if batch_size > 1:
                        # Change all other rows, including unused end slots;
                        # the first text must remain independent of its peers.
                        changed_ids = ids.copy()
                        changed_ids[1:] = (changed_ids[1:] + 5) % 16
                        changed = np.asarray(model.predict({"input_ids": changed_ids,
                                                           "attention_mask": mask})["embedding"], dtype=np.float64)
                        np.testing.assert_allclose(changed[0], output[0], atol=1e-5, rtol=1e-5,
                                                   err_msg="another batch row changed the first embedding")

    def test_source_fused_qkv_does_not_require_separate_projection_tensors(self):
        np, ct = self.np, self.ct
        reader, weights = self.tiny_reader()
        combined = {}
        for suffix in ("weight", "bias"):
            combined["blk.0.attn_qkv." + suffix] = np.concatenate([
                weights["blk.0." + name + "." + suffix] for name in ("attn_q", "attn_k", "attn_v")
            ]).astype(np.float32)
        reader.tensors = [tensor for tensor in reader.tensors
                          if not tensor.name.startswith(("blk.0.attn_q.", "blk.0.attn_k.", "blk.0.attn_v."))]
        reader.tensors.extend(types.SimpleNamespace(name=name, shape=value.shape[::-1], data=value,
                                                    tensor_type=self.gguf.GGMLQuantizationType.F32)
                              for name, value in combined.items())
        spec = read_spec(reader, 8)
        ids = np.array([[1, 2, 3, 15, 15, 15, 15, 15]], dtype=np.int32)
        mask = np.array([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int32)
        for prune_cls in (False, True):
            with self.subTest(prune_cls=prune_cls):
                program = build_program(reader, spec, prune_cls=prune_cls)
                convolutions = [op for op in program.functions["main"].operations if op.op_type == "conv"]
                self.assertEqual(len(convolutions), 4)
                self.assertEqual(convolutions[0].outputs[0].shape[-1], 8)
                self.assertEqual(convolutions[-1].outputs[0].shape[-1], 1 if prune_cls else 8)
                model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16,
                                   compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_ONLY)
                output = np.asarray(model.predict({"input_ids": ids, "attention_mask": mask})["embedding"],
                                    dtype=np.float64).reshape(-1)
                output /= np.linalg.norm(output)
                np.testing.assert_allclose(output, self.reference(weights, spec, ids[0, :3], mask[0, :3]),
                                           atol=3e-3, rtol=3e-3)

    def test_missing_projection_biases_default_to_zero(self):
        np, ct = self.np, self.ct
        ids = np.array([[1, 2, 3, 15, 15, 15, 15, 15]], dtype=np.int32)
        mask = np.array([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int32)
        for source_fused in (False, True):
            reader, weights = self.tiny_reader()
            missing = {f"blk.0.{name}.bias" for name in ("attn_q", "attn_k", "attn_v", "ffn_up")}
            reader.tensors = [tensor for tensor in reader.tensors if tensor.name not in missing]
            for name in missing:
                weights[name] = np.zeros_like(weights[name])
            if source_fused:
                # Leave out the optional combined bias as well as all separate
                # projection tensors. The NumPy reference remains separate.
                reader.tensors = [tensor for tensor in reader.tensors
                                  if not tensor.name.startswith(("blk.0.attn_q.", "blk.0.attn_k.", "blk.0.attn_v."))]
                combined = np.concatenate([weights[f"blk.0.{name}.weight"]
                                           for name in ("attn_q", "attn_k", "attn_v")]).astype(np.float32)
                reader.tensors.append(types.SimpleNamespace(name="blk.0.attn_qkv.weight", shape=combined.shape[::-1],
                                                            data=combined, tensor_type=self.gguf.GGMLQuantizationType.F32))
            spec = read_spec(reader, 8)
            for prune_cls in (False, True):
                with self.subTest(source_fused=source_fused, prune_cls=prune_cls):
                    program = build_program(reader, spec, prune_cls=prune_cls)
                    model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16,
                                       compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_ONLY)
                    output = np.asarray(model.predict({"input_ids": ids, "attention_mask": mask})["embedding"],
                                        dtype=np.float64).reshape(-1)
                    output /= np.linalg.norm(output)
                    np.testing.assert_allclose(output, self.reference(weights, spec, ids[0, :3], mask[0, :3]),
                                               atol=3e-3, rtol=3e-3)

    def test_palette8_changes_only_convolution_weights(self):
        np, ct = self.np, self.ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types as mil_types

        rng = np.random.default_rng(9287)
        kernel = (rng.normal(size=(96, 192, 1, 1)) * .05).astype(np.float16)
        bias = (rng.normal(size=96) * .1).astype(np.float16)
        # Larger than the projection bank: blanket compression would include
        # this vocabulary, despite its role being unrelated to convolution.
        vocabulary = rng.normal(size=(512, 96)).astype(np.float16)

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 192, 1, 4), dtype=mil_types.fp16),
                                 mb.TensorSpec(shape=(1,), dtype=mil_types.int32)], opset_version=ct.target.iOS16)
        def program(activations, ids):
            projected = mb.conv(x=activations, weight=kernel, bias=bias, name="projected")
            lookup = mb.gather(x=vocabulary, indices=ids, axis=0, name="lookup")
            return projected, lookup

        plain = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16,
                           compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_ONLY)
        names = conv_weight_names(plain.get_spec())
        self.assertEqual(len(names), 1)
        # Lower only the helper threshold in the fixture, preserving the
        # production 4096-element threshold and a meaningful >256-value bank.
        palette = exporter.compress_conv_weights(plain, "palette8", weight_threshold=0)
        blocks = next(iter(palette.get_spec().mlProgram.functions.values())).block_specializations
        operations = next(iter(blocks.values())).operations
        compressed = [op for op in operations if op.type.startswith("constexpr_")]
        self.assertEqual(len(compressed), 1, "palette compression reached a non-convolution constant")
        self.assertEqual(compressed[0].type, "constexpr_lut_to_dense")
        activations = rng.normal(size=(1, 192, 1, 4)).astype(np.float16)
        reference = plain.predict({"activations": activations, "ids": np.array([17], dtype=np.int32)})
        actual = palette.predict({"activations": activations, "ids": np.array([17], dtype=np.int32)})
        np.testing.assert_array_equal(actual["lookup"], reference["lookup"])
        left, right = actual["projected"].reshape(-1).astype(np.float64), reference["projected"].reshape(-1).astype(np.float64)
        self.assertTrue(np.isfinite(left).all())
        cosine = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
        self.assertGreater(cosine, .999)


if __name__ == "__main__":
    unittest.main()
