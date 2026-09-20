"""Materializes cmd/gopherllm-synth-testmodel's deterministic fixture once per
test session, so the binding tests exercise a real GGUF without a binary
model file checked into the repo."""
import os
import subprocess
import tempfile

import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@pytest.fixture(scope="session")
def synth_model_path() -> str:
    out = os.path.join(tempfile.gettempdir(), "gopherllm-python-binding-test-model.gguf")
    env = dict(os.environ, GO111MODULE="on")
    result = subprocess.run(
        ["go", "run", "./cmd/gopherllm-synth-testmodel", "-out", out],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "failed to run `go run ./cmd/gopherllm-synth-testmodel` -- is Go installed and on PATH?\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return out
