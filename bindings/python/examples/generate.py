#!/usr/bin/env python3
"""Run with: GOPHERLLM_LIB_DIR=../../build/capi python3 generate.py <model.gguf> "<prompt>" """
import sys

from gopherllm import Engine, GenerationOptions


def main() -> None:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <model.gguf> [prompt]", file=sys.stderr)
        raise SystemExit(1)
    model_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello!"

    with Engine() as engine:
        engine.load(model_path)
        info = engine.info()
        print(f"loaded {engine.model_name()} ({info.architecture})")

        print("streaming: ", end="")
        for delta in engine.stream(prompt, GenerationOptions(max_tokens=64)):
            print(delta, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
