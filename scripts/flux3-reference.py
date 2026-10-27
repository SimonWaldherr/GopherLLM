#!/usr/bin/env python3
"""Build small native FLUX 3 differential fixtures using BFL's actual modules.

Only test tooling uses Python. No pretrained weights are downloaded. Supply a
checkout of black-forest-labs/flux-3-image-klein at the pinned revision below.
Requires torch 2.8.0, transformers 4.57.1, tokenizers 0.22.2 and safetensors.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import Qwen3VLTextConfig, Qwen3VLTextModel

REVISION = "bfc8889a4497f846416f253a3972a1941ac698b9"
SOURCE_SHA256 = {
    "dit.py": "d4c0d2aea5282ae63ac32a1f3029a738ddab951bb1033d6f3a2e9bd192c89934",
    "autoencoder.py": "960e2a98046bcb78f2d36fbbd4848db726f14810c7887485f7e2682346f95189",
    "text_encoder.py": "3bce784f43b8e8d9eee0dab75d928006398efd80e77f9f179bba6ddd28ffe74b",
    "sampling.py": "22be0d823fb6a2c5bc993c77fe0836ca729219d34cc08e86d01a398e04bdcdf5",
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory containing official dit.py and autoencoder.py")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source_hashes = {}
    for filename, expected in SOURCE_SHA256.items():
        source_hashes[filename] = hashlib.sha256((args.source / filename).read_bytes()).hexdigest()
        if source_hashes[filename] != expected:
            parser.error(f"{filename} does not match BFL revision {REVISION}")
    torch.set_num_threads(1)
    torch.manual_seed(302611)
    dit = module("flux3_fixture_dit", args.source / "dit.py")
    ae = module("flux3_fixture_ae", args.source / "autoencoder.py")
    encoder_reference = module("qwen_fixture", Path(__file__).with_name("qwen-image21-encoder-reference.py"))
    dit.HIDDEN, dit.HEADS, dit.DEPTH, dit.DEPTH_SINGLE = 32, 4, 2, 3
    dit.AXES_DIM = (2, 2, 2, 2)
    dit.IN_CHANNELS = {"image": 128, "image_cond": 128, "text": 128}
    ae.CH = 32  # retain 32 groups and all production resampling/residual stages
    transformer = dit.DiT(flow_map=True).float().eval()
    autoencoder = ae.AutoEncoder().float().eval()
    cfg = Qwen3VLTextConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=36, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, vocab_size=300, max_position_embeddings=4096,
        rope_theta=5000000, rms_norm_eps=1e-6, attention_bias=False,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True})
    cfg._attn_implementation = "eager"
    encoder = Qwen3VLTextModel(cfg).float().eval()
    tokenizer_config = encoder_reference.tiny_tokenizer()
    tokenizer = Tokenizer.from_str(json.dumps(tokenizer_config))
    with torch.no_grad():
        for name, value in transformer.named_parameters():
            if "norm" in name:
                value.copy_(torch.linspace(0.7, 1.3, value.numel()).reshape_as(value))
        for name, value in encoder.named_parameters():
            if value.ndim == 1:
                value.copy_(torch.linspace(0.7, 1.3, value.numel()))
        encoder.norm.weight.copy_(torch.linspace(1.7, 2.3, 16))
        # Periodic nonzero VAE weights keep the compressed fixture small.
        # Distinct channel/spatial phases and nonuniform affine statistics
        # exercise normalization, padding, projection and patch ordering.
        for index, (name, value) in enumerate(autoencoder.named_parameters()):
            phase = torch.arange(value.numel()).reshape_as(value) % 251
            wave = torch.sin(phase.float() * 0.79 + index * 0.37)
            if "norm" in name and name.endswith("weight"):
                value.copy_(1 + 0.2 * wave)
            else:
                value.copy_(wave * (0.018 if value.ndim == 4 else 0.05))
        autoencoder.bn.running_mean.copy_(torch.linspace(-0.2, 0.3, 128))
        autoencoder.bn.running_var.copy_(torch.linspace(0.5, 1.5, 128))
    out = args.output
    (out / "text_encoder").mkdir(parents=True, exist_ok=True)
    save_file(transformer.state_dict(), out / "model.safetensors")
    vae_weights = {k: v for k, v in autoencoder.state_dict().items() if k.startswith(("decoder.", "bn."))}
    save_file(vae_weights, out / "ae.safetensors")
    raw = (out / "ae.safetensors").read_bytes()
    (out / "ae.safetensors.gz").write_bytes(gzip.compress(raw, mtime=0))
    (out / "ae.safetensors").unlink()
    save_file({"model.language_model." + k: v for k, v in encoder.state_dict().items()}, out / "text_encoder/model.safetensors")
    (out / "text_encoder/config.json").write_text(json.dumps({"model_type": "qwen3_vl", "text_config": cfg.to_dict()}, indent=2) + "\n")
    (out / "text_encoder/tokenizer.json").write_text(json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n")

    def encode(prompt):
        text = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
        ids = tokenizer.encode(text).ids[:4096]
        real = len(ids)
        length = min((real + 79) // 80 * 80, 4096)
        ids += [260] * (length - real)
        mask = torch.tensor([[1] * real + [0] * (length - real)])
        pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
        result = encoder(input_ids=torch.tensor([ids]), attention_mask=mask,
            position_ids=pos[None].expand(4, -1, -1), output_hidden_states=True, use_cache=False)
        features = torch.stack([result.hidden_states[k] for k in range(4, 33, 4)], dim=1)
        return features.permute(0, 2, 1, 3).reshape(length, -1).contiguous(), ids, real

    def pack(text, noise, h, w, time):
        ids = ae.latent_ids(h, w).float()
        ids[:, 1:3] *= 0.5
        text_ids = torch.zeros(len(text), 4)
        text_ids[:, 3] = torch.arange(len(text))
        return dit.Packed(
            data={"image": noise, "image_cond": torch.empty(0, 128), "text": text},
            ids={"image": ids, "image_cond": torch.empty(0, 4), "text": text_ids},
            timesteps={"image": torch.full((h*w,), time), "image_cond": torch.empty(0), "text": torch.zeros(len(text))},
            mode_indices={"image": [(0, h*w)], "image_cond": [(0, 0)], "text": [(0, len(text))]})

    reference = {"revision": REVISION, "source_sha256": source_hashes, "encoder": [], "transformer": [], "pipeline": []}
    tensors = {}
    with torch.inference_mode():
        for i, prompt in enumerate(["A blue cube.", "", "x" * 65, "café " * 30]):
            features, tokens, real = encode(prompt)
            key = f"encoder.{i}"
            tensors[key] = features
            reference["encoder"].append({"prompt": prompt, "tokens": tokens, "real": real, "key": key})
        for flow in (True, False):
            transformer.flow_map = flow
            for h, w in ((2, 3), (4, 5)):
                text = torch.randn(9, 128)
                noise = torch.randn(h*w, 128)
                for time, target in ((0.95, 0.8), (0.4, 0.0)):
                    key = f"transformer.{len(reference['transformer'])}"
                    tensors[key + ".text"], tensors[key + ".noise"] = text, noise
                    tensors[key + ".expected"] = transformer(pack(text, noise, h, w, time), torch.tensor([target]) if flow else None)
                    reference["transformer"].append({"h": h, "w": w, "flow": flow, "time": time, "target": target, "key": key})
        for flow, h, w in ((True, 4, 4), (True, 4, 6), (False, 4, 4)):
            transformer.flow_map = flow
            prompt = "A blue cube."
            text = encode(prompt)[0]
            uncond = encode("")[0]
            noise = torch.randn(h*w, 128)
            x = noise.clone()
            schedule = torch.linspace(1, 0, 3)
            schedule = 16*schedule/(1+15*schedule)
            for t, target in zip(schedule[:-1], schedule[1:]):
                velocity = transformer(pack(text, x, h, w, t.item()), target.reshape(1) if flow else None)
                if not flow:
                    u = transformer(pack(uncond, x, h, w, t.item()))
                    velocity = u + 4 * (velocity - u)
                x = x + (target-t)*velocity
            decoded = autoencoder.decode(x, ae.latent_ids(h, w))
            pixels = ((decoded.clamp(-1, 1)+1)*127.5).to(torch.uint8).permute(1, 2, 0).numpy()
            key = f"pipeline.{len(reference['pipeline'])}"
            tensors[key + ".noise"], tensors[key + ".latents"] = noise, x
            Image.fromarray(pixels).save(out / (key + ".png"))
            reference["pipeline"].append({"flow": flow, "h": h, "w": w, "prompt": prompt, "key": key})
        for steps in (1, 3, 16, 50):
            t = torch.linspace(1, 0, steps+1)
            tensors[f"schedule.{steps}"] = 16*t/(1+15*t)
    save_file({k: v.clone().contiguous() for k, v in tensors.items()}, out / "reference.safetensors")
    (out / "reference.json").write_text(json.dumps(reference, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
