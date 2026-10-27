#!/usr/bin/env python3
"""Download authorized FLUX 3 Image Klein weights as native Q8_0 components.

Accept BFL's model conditions on Hugging Face, then run `hf auth login` locally.
Python, numpy and huggingface_hub are used only for download/conversion.
Inference runs in Go. The shared range importer avoids a second full copy.
"""

import argparse
import functools
import json
import urllib.error
import urllib.parse
from pathlib import Path

from huggingface_hub import get_token

from download_qwen_image21 import fetch, import_shard

REPOSITORY = "black-forest-labs/flux-3-image-klein"
REVISION = "337337621b8526e120f2275d250242f4a6bedd75"


def needed(component, name):
    if component == "text_encoder":
        if name.startswith("model.language_model.layers."):
            return int(name.split(".")[3]) < 32
        return name == "model.language_model.embed_tokens.weight"
    if component == "vae":
        return name.startswith("decoder.") or name in ("bn.running_mean", "bn.running_var")
    # Reference-image branches are not needed for text-to-image.
    return not name.startswith(("emb_in.image_cond.", "content_mode_blocks.image_cond.",
                                "early_stream_modulations.image_cond.", "single_stream_modulations.image_cond."))


def download(directory, workers):
    token = get_token()
    if not token:
        raise ValueError("First accept the FLUX model conditions on Hugging Face and run `hf auth login` locally. Do not put tokens in command arguments.")
    get = functools.partial(fetch, token=token)
    base = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/"
    # Check access before creating files or downloading any large weights.
    license_text = get(base + "LICENSE.md")
    index = json.loads(get(base + "text_encoder/model.safetensors.index.json"))
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(Path(p).name != p or not p.endswith(".safetensors") for p in shards):
        raise ValueError("invalid text encoder shard index")
    provenance = {"repository": REPOSITORY, "revision": REVISION,
                  "matrices": "Q8_0", "runtime": "native Go", "layout": "BFL"}
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "source.json"
    if source.exists() and json.loads(source.read_text()) != provenance:
        raise ValueError("destination belongs to another checkpoint")
    source.write_text(json.dumps(provenance, indent=2) + "\n")
    (directory / "LICENSE.md").write_bytes(license_text)
    for name in ("NOTICE.md", "README.md", "config.json", "manifest.json", "text_encoder/config.json", "text_encoder/tokenizer.json", "text_encoder/tokenizer_config.json", "text_encoder/chat_template.json"):
        dest = directory / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(get(base + urllib.parse.quote(name)))
    for component, names in (("transformer", ["model.safetensors"]),
                             ("text_encoder", ["text_encoder/" + p for p in shards]),
                             ("vae", ["ae.safetensors"])):
        mapping = {}
        for name in names:
            dest = directory / Path(name).with_suffix(".gguf")
            tensors = import_shard(base + urllib.parse.quote(name), dest, component, REVISION, workers,
                include=functools.partial(needed, component), architecture="flux3_image_component", fetcher=get)
            for tensor in tensors:
                if tensor in mapping:
                    raise ValueError(f"duplicate tensor {tensor}")
                mapping[tensor] = dest.name
        if not mapping:
            raise ValueError(f"no {component} weights found")
        if component == "text_encoder":
            path = directory / "text_encoder/weights.index.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"weight_map": mapping}, indent=2) + "\n")
            tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=4)
    args = parser.parse_args()
    try:
        download(args.directory, args.workers)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            parser.exit(1, "Model access denied. Accept the model conditions at https://huggingface.co/" + REPOSITORY + " and run `hf auth login` locally.\n")
        raise
    except ValueError as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
