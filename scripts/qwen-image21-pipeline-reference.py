#!/usr/bin/env python3
"""Compose the small, official Qwen-Image-2.1 components into an eight-step oracle.

Test tooling only; Go inference does not run Python. Requires torch 2.8+,
diffusers 0.41.0, safetensors, numpy and Pillow. The conditioning features and
decoder weights come from the existing independently verified fixtures, so the
pipeline fixture does not duplicate their weights or require Transformers at
generation time. All weights are random, with no pretrained model downloads.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import diffusers
import numpy as np
import torch
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Transformer2DModel,
)
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AttnProcessor,
    QwenImage21KVCache,
)
from PIL import Image
from safetensors.torch import load_file, save_file

SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testdata", type=Path, default=Path("testdata"))
    args = parser.parse_args()
    root = args.testdata
    output = root / "qwen_image21_pipeline"
    (output / "transformer").mkdir(parents=True, exist_ok=True)
    (output / "scheduler").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(725901)
    encoder = json.loads((root / "qwen_image21_encoder.json").read_text())
    text = load_file(root / "qwen_image21_encoder.safetensors")[
        "reference.hidden"
    ].unsqueeze(0)
    cfg = {
        "patch_size": 1,
        "in_channels": 64,
        "out_channels": 64,
        "num_layers": 2,
        "attention_head_dim": 16,
        "num_attention_heads": 2,
        "context_in_dim": 16,
        "mlp_ratio": 3,
        "axes_dims_rope": [4, 6, 6],
        "eps": 1e-6,
        "causal_condition": True,
    }
    dit = QwenImage21Transformer2DModel(**cfg).eval()
    dit.set_attn_processor(QwenImage21AttnProcessor())
    with torch.no_grad():
        dit.txt_in.text_norm.weight.copy_(torch.randn(16) * 0.1)
        for block in dit.transformer_blocks:
            block.attn.norm_q.weight.copy_(0.9 + torch.rand(16) * 0.2)
            block.attn.norm_k.weight.copy_(0.9 + torch.rand(16) * 0.2)
    save_file(
        dit.state_dict(), output / "transformer" / "diffusion_pytorch_model.safetensors"
    )
    (output / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "QwenImage21Transformer2DModel", **cfg}, indent=2)
        + "\n"
    )

    vae_config = json.loads((root / "qwen_image21_vae_config.json").read_text())
    vae = AutoencoderKLQwenImage21.from_config(vae_config).float().eval()
    stored = load_file(root / "qwen_image21_vae.safetensors")
    decoded_weights = {
        k: v.float()
        for k, v in stored.items()
        if k.startswith(("decoder.", "post_quant_conv."))
    }
    loaded = vae.load_state_dict(decoded_weights, strict=False)
    if loaded.unexpected_keys or any(
        k.startswith(("decoder.", "post_quant_conv.")) for k in loaded.missing_keys
    ):
        raise ValueError("incomplete decoder fixture")
    scheduler_config = {
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "use_dynamic_shifting": False,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "base_image_seq_len": 256,
        "max_image_seq_len": 4096,
        "shift_terminal": None,
        "time_shift_type": "exponential",
    }
    (output / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(
            {"_class_name": "FlowMatchEulerDiscreteScheduler", **scheduler_config},
            indent=2,
        )
        + "\n"
    )
    index = {
        "_class_name": "QwenImage21Pipeline",
        "sample_sigmas": SIGMAS,
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLQwenImage21"],
    }
    (output / "model_index.json").write_text(json.dumps(index, indent=2) + "\n")
    cases = []
    with torch.inference_mode():
        for height, width in [(32, 64), (64, 64)]:
            h, w = height // 16, width // 16
            n = h * w
            initial = torch.randn(1, n, 64)
            latent = initial.clone()
            scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)
            scheduler.set_timesteps(sigmas=SIGMAS)
            scheduler.set_begin_index(0)
            cache = QwenImage21KVCache(cfg["num_layers"])
            mask = torch.cat(
                [
                    torch.zeros(1, text.shape[1], dtype=torch.bool),
                    torch.ones(1, n // 4, dtype=torch.bool),
                ],
                dim=1,
            )
            for step, time in enumerate(scheduler.timesteps):
                predicted = dit(
                    hidden_states=latent,
                    encoder_hidden_states=text,
                    timestep=(time / 1000).reshape(1),
                    img_shapes=[[(1, h, w)]],
                    img_mask=mask,
                    kv_cache=cache,
                    kv_cache_mode="extract" if step == 0 else "cached",
                ).sample[:, -n:]
                latent = scheduler.step(predicted, time, latent).prev_sample
            unpacked = latent.transpose(1, 2).reshape(1, 64, 1, h, w)
            mean = torch.tensor(vae_config["latents_mean"]).view(1, 64, 1, 1, 1)
            std = torch.tensor(vae_config["latents_std"]).view(1, 64, 1, 1, 1)
            rgba = vae.decode(unpacked * std + mean).sample[0, :, 0]
            pixels = (
                ((rgba.permute(1, 2, 0).numpy() / 2 + 0.5).clip(0, 1) * 255)
                .round()
                .astype(np.uint8)
            )
            name = f"{width}x{height}.png"
            Image.fromarray(pixels).save(output / name)
            cases.append(
                {
                    "width": width,
                    "height": height,
                    "image": name,
                    "noise": initial.flatten().tolist(),
                    "latents": latent.flatten().tolist(),
                }
            )
    metadata = {
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "prompt": encoder["prompt"],
        "sources": {
            k: hashlib.sha256(Path(inspect.getfile(v)).read_bytes()).hexdigest()
            for k, v in {
                "transformer": QwenImage21Transformer2DModel,
                "vae": AutoencoderKLQwenImage21,
                "scheduler": FlowMatchEulerDiscreteScheduler,
            }.items()
        },
        "inputs": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "qwen_image21_encoder.safetensors",
                "qwen_image21_vae.safetensors",
            )
        },
        "cases": cases,
    }
    (output / "reference.json").write_text(
        json.dumps(metadata, separators=(",", ":")) + "\n"
    )
    print(f"wrote {len(cases)} native-pipeline reference cases to {output}")


if __name__ == "__main__":
    main()
