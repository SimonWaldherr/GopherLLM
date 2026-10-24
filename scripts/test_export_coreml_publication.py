"""Exercise export publication without Core ML or model weights."""

from contextlib import ExitStack
import errno
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_coreml_embeddings as embeddings
import export_coreml_ffn as ffn


class FakeModel:
    def __init__(self):
        self.user_defined_metadata = {}

    def save(self, destination):
        Path(destination).mkdir()
        (Path(destination) / "weights").write_bytes(b"package")


class Publication(unittest.TestCase):
    def export(self, exporter, output, compiled, compiler):
        reader = types.SimpleNamespace(tensors=[types.SimpleNamespace(name="token_embd.weight", shape=(8, 16))])
        spec = types.SimpleNamespace(dimension=8, hidden_dimension=16, layers=2, pooling=2,
                                     architecture="qwen35", layer=0, tokens=32)
        coreml = types.SimpleNamespace(
            convert=lambda *args, **kwargs: FakeModel(),
            target=types.SimpleNamespace(iOS16=16),
            precision=types.SimpleNamespace(FLOAT16="fp16", FLOAT32="fp32"),
            ComputeUnit=types.SimpleNamespace(CPU_AND_NE="ane"),
            models=types.SimpleNamespace(utils=types.SimpleNamespace(compile_model=compiler)))
        with ExitStack() as mocks:
            mocks.enter_context(patch.dict(sys.modules, {"coremltools": coreml,
                                                        "gguf": types.SimpleNamespace(GGUFReader=lambda _: reader)}))
            mocks.enter_context(patch.object(exporter, "read_spec", return_value=spec))
            mocks.enter_context(patch.object(exporter, "build_program", return_value=object()))
            if exporter is ffn:
                mocks.enter_context(patch.object(ffn, "source_fingerprints", return_value=[]))
                return exporter.export("source.gguf", output, compile_to=compiled)
            return exporter.export("source.gguf", output, 32, compile_to=compiled)

    def test_compiler_failure_does_not_publish_or_leave_staging_files(self):
        def fail(source, destination_path):
            Path(destination_path).mkdir()
            raise OSError("compiler failed")

        for exporter in (embeddings, ffn):
            with self.subTest(exporter=exporter.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output, compiled = root / "package" / "encoder.mlpackage", root / "compiled" / "encoder.mlmodelc"
                with self.assertRaisesRegex(OSError, "compiler failed"):
                    self.export(exporter, output, compiled, fail)
                self.assertEqual(list(output.parent.iterdir()), [])
                self.assertEqual(list(compiled.parent.iterdir()), [])

    def test_separate_destination_filesystems(self):
        def compile_model(source, destination_path):
            self.assertTrue((Path(source) / "weights").is_file())
            Path(destination_path).mkdir()
            (Path(destination_path) / "compiled").write_bytes(b"compiled")

        rename = Path.rename
        for exporter in (embeddings, ffn):
            with self.subTest(exporter=exporter.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output, compiled = root / "volume-a" / "encoder.mlpackage", root / "volume-b" / "encoder.mlmodelc"

                def same_volume_rename(source, destination):
                    # Treat the two output parents as separate volumes. Staged
                    # artifacts must be created beneath their own destination.
                    if source.relative_to(root).parts[0] != destination.relative_to(root).parts[0]:
                        raise OSError(errno.EXDEV, "cross-device rename")
                    return rename(source, destination)

                with patch.object(Path, "rename", same_volume_rename):
                    result = self.export(exporter, output, compiled, compile_model)
                self.assertEqual(result["compiled_model"], str(compiled))
                self.assertEqual((output / "weights").read_bytes(), b"package")
                self.assertEqual((compiled / "compiled").read_bytes(), b"compiled")
                if exporter is ffn:
                    self.assertEqual(json.loads(compiled.with_suffix(".json").read_text())["model"], compiled.name)
                self.assertFalse(list(root.glob("*/.gopherllm-*")))

    def test_destination_created_during_compilation_is_preserved(self):
        for exporter in (embeddings, ffn):
            with self.subTest(exporter=exporter.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output, compiled = root / "encoder.mlpackage", root / "encoder.mlmodelc"

                def compile_model(source, destination_path):
                    Path(destination_path).mkdir()
                    output.mkdir()
                    (output / "existing").write_bytes(b"keep")

                with self.assertRaisesRegex(ValueError, "already exists"):
                    self.export(exporter, output, compiled, compile_model)
                self.assertEqual((output / "existing").read_bytes(), b"keep")
                self.assertFalse(compiled.exists())
                self.assertFalse(list(root.glob("*.json")))
                self.assertFalse(list(root.glob(".gopherllm-*")))


if __name__ == "__main__":
    unittest.main()
