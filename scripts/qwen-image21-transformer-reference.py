#!/usr/bin/env python3
"""Regenerate tiny Qwen-Image-2.1 DiT and scheduler differential fixtures.

Test tooling only: inference in GopherLLM does not use Python. Requires torch
2.8+, diffusers 0.41.0 and safetensors. Uses randomly initialized official
Diffusers modules and downloads no pretrained weights.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import diffusers
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AttnProcessor,
    QwenImage21KVCache,
)
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(49285)
    cfg = {
        "patch_size": 1,
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 2,
        "attention_head_dim": 16,
        "num_attention_heads": 2,
        "context_in_dim": 24,
        "mlp_ratio": 3,
        "axes_dims_rope": [4, 6, 6],
        "eps": 1e-6,
        "causal_condition": True,
    }
    model = QwenImage21Transformer2DModel(**cfg).eval()
    model.set_attn_processor(QwenImage21AttnProcessor())
    # Nonzero zero-centred text weights and unequal Q/K norms exercise their
    # exact contracts rather than relying on the initialization identities.
    with torch.no_grad():
        model.txt_in.text_norm.weight.copy_(torch.randn(24) * 0.1)
        for block in model.transformer_blocks:
            block.attn.norm_q.weight.copy_(0.9 + torch.rand(16) * 0.2)
            block.attn.norm_k.weight.copy_(0.9 + torch.rand(16) * 0.2)
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), args.output / "diffusion_pytorch_model.safetensors")
    (args.output / "config.json").write_text(
        json.dumps({"_class_name": "QwenImage21Transformer2DModel", **cfg}, indent=2)
        + "\n"
    )
    cases = []
    with torch.inference_mode():
        for height, width, prefix in [(2, 2, 7), (2, 4, 3), (3, 4, 5)]:
            n = height * width
            text = torch.randn(1, prefix, cfg["context_in_dim"])
            img_mask = torch.cat(
                [
                    torch.zeros(1, prefix, dtype=torch.bool),
                    torch.ones(1, n // 4, dtype=torch.bool),
                ],
                dim=1,
            )
            cache = QwenImage21KVCache(cfg["num_layers"])
            steps = []
            for i, time in enumerate([0.95, 0.65, 0.2]):
                latent = torch.randn(1, n, cfg["in_channels"])
                kwargs = {
                    "hidden_states": latent,
                    "encoder_hidden_states": text,
                    "timestep": torch.tensor([time]),
                    "img_shapes": [[(1, height, width)]],
                    "img_mask": img_mask,
                }
                full = model(**kwargs).sample[:, -n:]
                cached = model(
                    **kwargs,
                    kv_cache=cache,
                    kv_cache_mode="extract" if i == 0 else "cached",
                ).sample[:, -n:]
                torch.testing.assert_close(cached, full, rtol=2e-5, atol=2e-6)
                steps.append(
                    {
                        "time": time,
                        "latent": latent.flatten().tolist(),
                        "expected": full.flatten().tolist(),
                    }
                )
            cases.append(
                {
                    "height": height,
                    "width": width,
                    "text": text.flatten().tolist(),
                    "steps": steps,
                }
            )
    base_config = {
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "use_dynamic_shifting": True,
        "base_shift": 0.5,
        "max_shift": 0.9,
        "base_image_seq_len": 256,
        "max_image_seq_len": 8192,
        "shift_terminal": 0.02,
        "time_shift_type": "exponential",
    }
    schedules = []
    for tokens, steps in [(256, 4), (1024, 40), (16384, 40)]:
        scheduler = FlowMatchEulerDiscreteScheduler(**base_config)
        mu = 0.5 + (tokens - 256) * (0.9 - 0.5) / (8192 - 256)
        scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps), mu=mu)
        schedules.append(
            {"tokens": tokens, "steps": steps, "expected": scheduler.sigmas.tolist()}
        )
    metadata = {
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "source_sha256": hashlib.sha256(
            Path(inspect.getfile(QwenImage21Transformer2DModel)).read_bytes()
        ).hexdigest(),
        "cases": cases,
        "schedules": schedules,
    }
    (args.output / "reference.json").write_text(
        json.dumps(metadata, separators=(",", ":")) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in metadata.items() if k not in ("cases", "schedules")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
