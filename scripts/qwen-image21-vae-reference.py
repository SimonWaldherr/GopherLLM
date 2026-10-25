#!/usr/bin/env python3
"""Generate small offline Qwen-Image-2.1 decoder differential fixtures.

Test tooling only. Requires torch==2.8.0, safetensors==0.6.2, numpy and Pillow.
The official Diffusers decoder is evaluated at reduced channel width with
random weights; this checks graph/math compatibility, not pretrained quality.
Pass the locally downloaded, pinned upstream source; no model is downloaded.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

COMMIT = "aaf57d03a3edfbd9de457d2872417a92f149a91d"
SOURCE_SHA256 = "1e29f252dbddd044b84c8b794589001ce94bb125a8f13eb5b226ef30c1e2ea11"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="autoencoder_kl_qwenimage21.py at the pinned Diffusers commit",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("testdata"))
    parser.add_argument("--weight-dtype", choices=("f32", "bf16"), default="bf16")
    args = parser.parse_args()
    source = args.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("upstream source hash differs from the pinned decoder")
    tree = ast.parse(source)
    # Evaluate only the upstream component classes, without importing the
    # whole Diffusers package or model-loading hooks. Their sole dependencies
    # are torch, SiLU and CACHE_T.
    components = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name.startswith("QwenImage21")
        ],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "CACHE_T": 2,
        "get_activation": lambda name: torch.nn.SiLU() if name == "silu" else None,
    }
    exec(compile(components, str(args.source), "exec"), namespace)  # noqa: S102 — pinned source hash verified above
    torch.manual_seed(81231)
    torch.set_num_threads(1)
    decoder = namespace["QwenImage21Decoder3d"](
        dim=2,
        z_dim=64,
        dim_mult=[1, 2, 4, 8, 8],
        num_res_blocks=2,
        temperal_upsample=[True, True, True, False],
        out_channels=4,
        is_residual=True,
    ).eval()
    post_quant = namespace["QwenImage21CausalConv3d"](64, 64, 1).eval()
    storage_dtype = torch.bfloat16 if args.weight_dtype == "bf16" else torch.float32
    # Native inference expands stored BF16 values and uses F32 arithmetic.
    # Match that arithmetic, while saving Turbo's BF16 storage representation.
    decoder.to(storage_dtype).float()
    post_quant.to(storage_dtype).float()
    conv_count = sum(
        isinstance(m, namespace["QwenImage21CausalConv3d"]) for m in decoder.modules()
    )
    weights = {
        "decoder." + k: v.to(storage_dtype).contiguous()
        for k, v in decoder.state_dict().items()
    }
    weights.update(
        {
            "post_quant_conv." + k: v.to(storage_dtype).contiguous()
            for k, v in post_quant.state_dict().items()
        }
    )
    mean = torch.arange(64, dtype=torch.float32).sin() * 0.2
    std = 1 + torch.arange(64, dtype=torch.float32) * 0.01
    config = {
        "_class_name": "AutoencoderKLQwenImage21",
        "base_dim": 2,
        "decoder_base_dim": 2,
        "z_dim": 64,
        "dim_mult": [1, 2, 4, 8, 8],
        "num_res_blocks": 2,
        "in_channels": 4,
        "out_channels": 4,
        "is_residual": True,
        "patch_size": None,
        "scale_factor_spatial": 16,
        "temperal_downsample": [False, True, True, True],
        "latents_mean": mean.tolist(),
        "latents_std": std.tolist(),
        "attn_scales": [],
        "dropout": 0.0,
    }

    def decode_tile(z):
        return decoder(
            post_quant(z),
            feat_cache=[None] * conv_count,
            feat_idx=[0],
            first_chunk=True,
        )

    def decode(z):
        h, w = z.shape[-2:]
        if h <= 16 and w <= 16:
            return decode_tile(z).clamp(-1, 1)
        # Follow AutoencoderKLQwenImage21.tiled_decode's exact blend order.
        rows = []
        for top in range(0, h, 12):
            row = []
            for left in range(0, w, 12):
                row.append(decode_tile(z[:, :, :, top : top + 16, left : left + 16]))
            rows.append(row)
        result = []
        for i, row in enumerate(rows):
            outputs = []
            for j, tile in enumerate(row):
                if i:
                    a = rows[i - 1][j]
                    extent = min(a.shape[-2], tile.shape[-2], 64)
                    for y in range(extent):
                        tile[:, :, :, y, :] = a[:, :, :, -extent + y, :] * (
                            1 - y / extent
                        ) + tile[:, :, :, y, :] * (y / extent)
                if j:
                    a = row[j - 1]
                    extent = min(a.shape[-1], tile.shape[-1], 64)
                    for x in range(extent):
                        tile[:, :, :, :, x] = a[:, :, :, :, -extent + x] * (
                            1 - x / extent
                        ) + tile[:, :, :, :, x] * (x / extent)
                outputs.append(tile[:, :, :, :192, :192])
            result.append(torch.cat(outputs, dim=-1))
        return torch.cat(result, dim=3)[:, :, :, : h * 16, : w * 16].clamp(-1, 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for name, h, w in (("small", 2, 3), ("tiled", 17, 13)):
            latents = torch.randn(1, 64, 1, h, w) * 0.35
            weights[name + ".latents"] = (
                latents.flatten(2).transpose(1, 2).squeeze(0).contiguous()
            )
            z = latents * std.view(1, 64, 1, 1, 1) + mean.view(1, 64, 1, 1, 1)
            decoded = decode(z)[0, :, 0]
            if name == "small":
                weights[name + ".output"] = decoded.contiguous()
            pixels = (
                ((decoded.permute(1, 2, 0).numpy() / 2 + 0.5).clip(0, 1) * 255)
                .round()
                .astype(np.uint8)
            )
            Image.fromarray(pixels).save(
                args.output_dir / ("qwen_image21_vae_" + name + ".png")
            )
    save_file(
        weights,
        args.output_dir / "qwen_image21_vae.safetensors",
        metadata={
            "diffusers_commit": COMMIT,
            "source_sha256": SOURCE_SHA256,
            "torch": torch.__version__,
            "weight_dtype": args.weight_dtype,
            "kind": "random reduced-width decoder",
        },
    )
    (args.output_dir / "qwen_image21_vae_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
