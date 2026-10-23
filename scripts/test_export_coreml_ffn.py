"""FFN contracts run without Core ML; native graph parity is opt-in on macOS."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import hashlib
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_coreml_ffn as exporter


class Field:
    def __init__(self, value):
        self.value = value

    def contents(self):
        return self.value


def fixture(architecture="qwen35", **changes):
    values = {"general.architecture": architecture, architecture + ".block_count": 2,
              architecture + ".embedding_length": 8, architecture + ".feed_forward_length": 16}
    values.update(changes)
    tensors = []
    for layer in range(2):
        for name, rows, columns in (("gate", 16, 8), ("up", 16, 8), ("down", 8, 16)):
            tensors.append(types.SimpleNamespace(name=f"blk.{layer}.ffn_{name}.weight",
                                                 shape=(columns, rows), data=b"encoded", tensor_type=12))
    return types.SimpleNamespace(fields={name: Field(value) for name, value in values.items()}, tensors=tensors)


class Contract(unittest.TestCase):
    def test_supported_dense_architectures_and_layer_selection(self):
        for arch in sorted(exporter.ARCHITECTURES):
            with self.subTest(architecture=arch):
                spec = exporter.read_spec(fixture(arch), 1, 32)
                self.assertEqual(spec, exporter.FFNSpec(arch, 1, 8, 16, 32))
        spec = exporter.read_spec(fixture(**{"qwen35.feed_forward_length": [16, 16]}), 1, 1)
        self.assertEqual(spec.hidden_dimension, 16)

    def test_rejects_invalid_geometry_before_optional_imports(self):
        for layer, tokens in ((True, 1), (-1, 1), (2, 1), (0, True), (0, 0), (0, 129), (0, 1.0)):
            with self.subTest(layer=layer, tokens=tokens), self.assertRaises(ValueError):
                exporter.read_spec(fixture(), layer, tokens)
        for layer, tokens in ((True, 1), (-1, 1), (0, True), (0, 0), (0, 129)):
            with self.subTest(layer=layer, tokens=tokens), self.assertRaises(ValueError):
                exporter.export(None, None, layer=layer, tokens=tokens)
        for changes in ({"general.architecture": "qwen"}, {"general.architecture": "qwen35moe"}, {"qwen35.expert_count": 8},
                        {"qwen35.expert_used_count": 2}, {"qwen35.embedding_length": 0},
                        {"qwen35.block_count": True}, {"qwen35.feed_forward_length": [16]},
                        {"qwen35.feed_forward_length": float("nan")},
                        {"qwen35.embedding_length": 16384}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                exporter.read_spec(fixture(**changes), 0, 128)

    def test_rejects_missing_mismatched_routed_fused_and_biased_tensors(self):
        for action in ("missing", "shape", "bias", "fused", "route", "duplicate"):
            reader = fixture()
            first = reader.tensors[0]
            if action == "missing":
                reader.tensors.pop(0)
            elif action == "shape":
                first.shape = (16, 8)
            elif action == "bias":
                reader.tensors.append(types.SimpleNamespace(name="blk.0.ffn_gate.bias"))
            elif action == "fused":
                reader.tensors.append(types.SimpleNamespace(name="blk.0.ffn_gate_up.weight"))
            elif action == "route":
                reader.tensors.append(types.SimpleNamespace(name="blk.0.ffn_gate_inp.weight"))
            else:
                reader.tensors.append(first)
            with self.subTest(action=action), self.assertRaises(ValueError):
                exporter.read_spec(reader, 0, 32)

    def test_only_selected_source_bytes_are_hashed(self):
        reader = fixture()
        reader.tensors[3].data = object()  # another layer must never be read
        spec = exporter.read_spec(reader, 0, 32)
        fingerprints = exporter.source_fingerprints(reader, spec)
        self.assertEqual([value["name"] for value in fingerprints],
                         ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"])
        self.assertEqual(fingerprints[0], {"name": "blk.0.ffn_gate.weight", "shape": [16, 8],
                                          "tensor_type": 12, "sha256": hashlib.sha256(b"encoded").hexdigest()})
        contract = exporter.manifest(spec, fingerprints, "none", "block.mlmodelc")
        self.assertEqual(contract["version"], 1)
        self.assertEqual(contract["input"], "hidden")
        self.assertEqual(contract["output"], "output")
        self.assertEqual(contract["compute_precision"], "float16")
        self.assertEqual(contract["activation"], "silu_exp")
        self.assertEqual(contract["activation_scale"], 16)
        self.assertNotIn("source", contract)

    def test_destination_and_compression_validation_before_optional_imports(self):
        for scale in (None, True, 0, -1, 3, 8192, 16.0, "16"):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "activation scale"):
                exporter.export(None, None, activation_scale=scale)
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "activation scale"):
                exporter.build_program(None, None, activation_scale=scale)
        for hidden_chunk in (None, True, -1, 16385, 2.0, "8192"):
            with self.subTest(hidden_chunk=hidden_chunk), self.assertRaisesRegex(ValueError, "hidden chunk"):
                exporter.export(None, None, hidden_chunk=hidden_chunk)
            with self.subTest(hidden_chunk=hidden_chunk), self.assertRaisesRegex(ValueError, "hidden chunk"):
                exporter.build_program(None, None, hidden_chunk=hidden_chunk)
        for mode in (None, True, "linear8", "palette4"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "compression"):
                exporter.export("model.gguf", "new.mlpackage", weight_compression=mode)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in (".mlmodel", ".json", ""):
                with self.assertRaisesRegex(ValueError, "mlpackage"):
                    exporter.export("missing", root / ("model" + suffix))
            for name in ("model.mlpackage", "model.json", "compiled.mlmodelc", "compiled.json"):
                path = root / name
                path.touch()
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "already exists"):
                    exporter.export("missing", root / "model.mlpackage", compile_to=root / "compiled.mlmodelc")
                path.unlink()

    def test_cli_forwards_explicit_flags_and_defaults(self):
        cases = (([], {"layer": 0, "tokens": 32, "compile_to": None, "weight_compression": "none", "hidden_chunk": 0, "activation_scale": 16}),
                 (["--layer", "1", "--tokens", "16", "--weight-compression", "palette8", "--compile-to", "block.mlmodelc", "--hidden-chunk", "8", "--activation-scale", "4"],
                  {"layer": 1, "tokens": 16, "compile_to": "block.mlmodelc", "weight_compression": "palette8", "hidden_chunk": 8, "activation_scale": 4}))
        for arguments, keywords in cases:
            with self.subTest(arguments=arguments), \
                    patch.object(sys, "argv", ["export_coreml_ffn.py", "model.gguf", "block.mlpackage", *arguments]), \
                    patch.object(exporter, "export", return_value={}) as export_mock, redirect_stdout(StringIO()):
                exporter.main()
                export_mock.assert_called_once_with("model.gguf", "block.mlpackage", **keywords)

    def test_cli_reports_invalid_arguments_without_traceback(self):
        for arguments in (["--tokens", "129"], ["--layer", "-1"], ["--weight-compression", "palette4"]):
            with self.subTest(arguments=arguments), \
                    patch.object(sys, "argv", ["export_coreml_ffn.py", "missing", "new.mlpackage", *arguments]), \
                    redirect_stderr(StringIO()) as stderr, self.assertRaises(SystemExit):
                exporter.main()
            self.assertNotIn("Traceback", stderr.getvalue())


@unittest.skipUnless(os.environ.get("GOPHERLLM_COREML_EXPORT_TESTS") == "1" and sys.platform == "darwin",
                     "set GOPHERLLM_COREML_EXPORT_TESTS=1 on macOS for native graph parity")
class GraphParity(unittest.TestCase):
    def test_projection_scale_rejects_fp16_weight_overflow(self):
        import numpy as np
        import gguf
        for overflow_row in (0, 200):
            with self.subTest(row=overflow_row):
                reader = fixture(**{"qwen35.feed_forward_length": 256})
                for tensor in reader.tensors:
                    tensor.shape = (256, 8) if "down" in tensor.name else (8, 256)
                    tensor.data = np.full(tensor.shape[::-1], .01, dtype=np.float32)
                    tensor.tensor_type = gguf.GGMLQuantizationType.F32
                reader.tensors[0].data[overflow_row, 0] = 5000
                original = reader.tensors[0].data.copy()
                spec = exporter.read_spec(reader, 0, 1)
                with self.assertRaisesRegex(ValueError, "activation scale overflows"):
                    exporter.build_program(reader, spec, activation_scale=16)
                np.testing.assert_array_equal(reader.tensors[0].data, original)

    def test_stable_silu_matches_real_gate_range_on_cpu_and_neural_engine(self):
        import numpy as np
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types

        # Real Ministral gates occupy this interval. Core ML's Swish op has
        # about 5% relative L2 error on the M2 Max ANE for this distribution.
        # Include extremes to catch exp(-x) overflow in a naive decomposition.
        values = np.linspace(-2, 2, 8192, dtype=np.float32).reshape(32, 256)
        values[0, :8] = [-80, -20, -12, -0.0, 0.0, 12, 20, 80]

        @mb.program(input_specs=[mb.TensorSpec(shape=(32, 256), dtype=types.fp32)],
                    opset_version=ct.target.iOS16)
        def program(hidden):
            x = mb.reshape(x=mb.transpose(x=mb.cast(x=hidden, dtype="fp16"), perm=[1, 0]),
                           shape=[1, 256, 1, 32])
            result = exporter.stable_silu(mb, np, x)
            return mb.cast(x=mb.transpose(x=mb.reshape(x=result, shape=[256, 32]), perm=[1, 0]),
                           dtype="fp32", name="output")

        model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16,
                           compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_ONLY)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "stable.mlpackage")
            model.save(path)
            expected = values.astype(np.float64) / (1 + np.exp(-values.astype(np.float64)))
            for units in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_NE):
                with self.subTest(units=units):
                    prediction = ct.models.MLModel(path, compute_units=units).predict({"hidden": values})["output"]
                    self.assertTrue(np.isfinite(prediction).all())
                    relative = np.linalg.norm(prediction.astype(np.float64) - expected) / np.linalg.norm(expected)
                    self.assertLess(relative, .003)
                    np.testing.assert_allclose(prediction[0, :8], expected[0, :8], atol=2e-5, rtol=.003)

    def test_fp16_swiglu_matches_independent_numpy_and_preserves_rows(self):
        import numpy as np
        import coremltools as ct
        import gguf
        reader = fixture()
        rng = np.random.default_rng(81734)
        weights = {}
        for tensor in reader.tensors:
            value = (rng.normal(size=tensor.shape[::-1]) * .15).astype(np.float32)
            tensor.data = value
            tensor.tensor_type = gguf.GGMLQuantizationType.F32
            weights[tensor.name] = value.astype(np.float16).astype(np.float64)
        for tokens, layer, hidden_chunk, activation_scale in ((1, 0, 0, 1), (7, 1, 0, 16), (32, 0, 0, 16),
                                                            (1, 1, 7, 16), (7, 0, 7, 1), (32, 1, 7, 16)):
            with self.subTest(tokens=tokens, layer=layer, hidden_chunk=hidden_chunk, activation_scale=activation_scale):
                spec = exporter.read_spec(reader, layer, tokens)
                program = exporter.build_program(reader, spec, hidden_chunk=hidden_chunk, activation_scale=activation_scale)
                self.assertEqual(tuple(program.functions["main"].inputs["hidden"].shape), (tokens, 8))
                self.assertEqual(tuple(program.functions["main"].outputs[0].shape), (tokens, 8))
                chunks = (spec.hidden_dimension + hidden_chunk - 1) // hidden_chunk if hidden_chunk else 1
                self.assertEqual(sum(op.op_type == "conv" for op in program.functions["main"].operations), 3 * chunks)
                if hidden_chunk:
                    hidden_ops = [op for op in program.functions["main"].operations
                                  if op.op_type == "conv" and not op.name.startswith("down_projection")]
                    self.assertTrue(all(op.outputs[0].shape[1] <= hidden_chunk for op in hidden_ops))
                model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16,
                                   compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_ONLY)
                from coremltools.proto import MIL_pb2
                main = model.get_spec().mlProgram.functions["main"]
                block = next(iter(main.block_specializations.values()))
                final = [op for op in block.operations if any(out.name == "output" for out in op.outputs)]
                self.assertEqual(len(final), 1)
                self.assertEqual(final[0].outputs[0].type.tensorType.dataType, MIL_pb2.FLOAT32)
                if activation_scale != 1:
                    self.assertEqual(final[0].type, "mul")
                    source = final[0].inputs["x"].arguments[0].name
                    producer = next(op for op in block.operations if any(out.name == source for out in op.outputs))
                    self.assertEqual(producer.outputs[0].type.tensorType.dataType, MIL_pb2.FLOAT32)
                self.assertTrue(all(op.outputs[0].type.tensorType.dataType == MIL_pb2.FLOAT16
                                    for op in block.operations if op.type == "conv"))
                x = (rng.normal(size=(tokens, 8)) * .3).astype(np.float32)
                x[0] = 0
                fp16x = x.astype(np.float16).astype(np.float64)
                gate = fp16x @ weights[f"blk.{layer}.ffn_gate.weight"].T
                up = fp16x @ weights[f"blk.{layer}.ffn_up.weight"].T
                expected = (gate / (1 + np.exp(-gate)) * up) @ weights[f"blk.{layer}.ffn_down.weight"].T
                result = np.asarray(model.predict({"hidden": x})["output"])
                self.assertEqual(result.shape, (tokens, 8))
                np.testing.assert_allclose(result, expected, atol=2e-5, rtol=5e-3)
                np.testing.assert_array_equal(result[0], np.zeros(8))
                if tokens > 1:
                    changed = x.copy()
                    changed[-1] += 1
                    other = np.asarray(model.predict({"hidden": changed})["output"])
                    np.testing.assert_array_equal(other[:-1], result[:-1])


if __name__ == "__main__":
    unittest.main()
