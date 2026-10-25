#!/usr/bin/env python3
"""Regenerate the offline DDPM-CIFAR reference with PyTorch and Diffusers.

Test tooling only; GopherLLM inference has no Python dependency. Requires
Python >=3.11, torch==2.8.0, diffusers==0.35.1, safetensors==0.6.2. Pass an
existing local google/ddpm-cifar10-32 directory; no model is downloaded.
"""

import argparse
import hashlib
import json
from pathlib import Path

import diffusers
import torch
from diffusers import DDIMScheduler, UNet2DModel
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = UNet2DModel.from_pretrained(
        args.model, local_files_only=True, low_cpu_mem_usage=False
    ).eval()
    scheduler = DDIMScheduler.from_pretrained(args.model, local_files_only=True)
    sample = torch.randn((1, 3, 32, 32), generator=torch.Generator().manual_seed(42))
    tensors = {"input": sample, "alphas": scheduler.alphas_cumprod}
    with torch.inference_mode():
        for timestep in (0, 500, 999):
            tensors[f"noise_{timestep}"] = model(sample, timestep).sample
        for steps in (4, 7, 23, 50):
            scheduler.set_timesteps(steps)
            image = sample.clone()
            for timestep in scheduler.timesteps:
                noise = model(image, timestep).sample
                image = scheduler.step(noise, timestep, image, eta=0).prev_sample
            tensors[f"ddim_{steps}"] = image
    checkpoint = args.model / "diffusion_pytorch_model.safetensors"
    with checkpoint.open("rb") as file:
        checkpoint_hash = hashlib.file_digest(file, "sha256").hexdigest()
    metadata = {
        "model": "google/ddpm-cifar10-32",
        "checkpoint_sha256": checkpoint_hash,
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "scheduler": "DDIM eta=0, leading spacing, epsilon prediction, clip_sample=true",
        "device": "cpu",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata=metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
