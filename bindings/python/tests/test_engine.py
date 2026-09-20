"""Exercises the whole FFI round trip against a real GGUF generated on the
fly by conftest.py's synth_model_path fixture (via
`go run ./cmd/gopherllm-synth-testmodel`), so this proves the package
actually links and calls into libgopherllm, not just that it imports -- with
no binary model checked into the repo. Run with:
  GOPHERLLM_LIB_DIR=<repo>/build/capi python3 -m pytest bindings/python/tests
"""
import pytest

from gopherllm import Engine, GenerationOptions, GopherLLMError, LoadOptions

EXPECTED_TEXT = '$(F=bBz".C\\[}3WeO`>i["cZ'


@pytest.fixture()
def loaded_engine(synth_model_path):
    engine = Engine()
    engine.load(synth_model_path)
    yield engine
    engine.close()


def test_load_reports_name_and_info(loaded_engine):
    assert loaded_engine.is_loaded()
    assert loaded_engine.model_name() == "gopherllm-synth-testmodel"
    info = loaded_engine.info()
    assert info.architecture == "llama"
    assert info.vocab_size == 98


def test_generate_returns_expected_deterministic_text(loaded_engine):
    text = loaded_engine.generate("Hallo", GenerationOptions(max_tokens=24, seed=1))
    assert text == EXPECTED_TEXT


def test_generate_stream_delivers_deltas_and_matching_result(loaded_engine):
    deltas = []
    result = loaded_engine.generate_stream(
        "Hallo", GenerationOptions(max_tokens=24, seed=1), on_delta=deltas.append
    )
    assert len(deltas) == 24, "every token is single-byte-ASCII, so one delta per token"
    assert "".join(deltas) == EXPECTED_TEXT
    assert result.finish_reason == "length"
    assert result.generated_tokens == 24


def test_stream_generator_yields_the_same_deltas(loaded_engine):
    deltas = list(loaded_engine.stream("Hallo", GenerationOptions(max_tokens=24, seed=1)))
    assert "".join(deltas) == EXPECTED_TEXT


def test_double_load_is_rejected(loaded_engine, synth_model_path):
    with pytest.raises(GopherLLMError, match="(?i)already loaded"):
        loaded_engine.load(synth_model_path)


def test_unload_then_reload(synth_model_path):
    engine = Engine()
    engine.load(synth_model_path)
    engine.unload()
    assert not engine.is_loaded()
    engine.load(synth_model_path, LoadOptions(threads=1))
    assert engine.is_loaded()
    engine.close()


def test_closed_engine_rejects_calls(synth_model_path):
    engine = Engine()
    engine.close()
    assert not engine.is_loaded()
    with pytest.raises(GopherLLMError, match="closed"):
        engine.load(synth_model_path)
    engine.close()  # idempotent
