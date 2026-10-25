#!/usr/bin/env python3
"""Write a local PyTorch HiFiGAN reference; never fetch model files.

Requires the optional development-only Python >=3.11, torch==2.8.0,
transformers==4.57.1 and safetensors==0.6.2 packages. The Go inference engine
does not execute or depend on this script.

For the small CI fixture with random weights instead of a trained model:
    python scripts/tts_vocoder_reference.py tiny testdata/tts_vocoder.safetensors
"""
import argparse
import json
from pathlib import Path

import safetensors
import torch
import transformers
from safetensors.torch import load_file, save_file
from transformers import VitsConfig
from transformers.models.vits.modeling_vits import VitsHifiGan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="local MMS model directory, or 'tiny' for the CI fixture")
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=7)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(41)
    tiny = args.model == "tiny"
    if tiny:
        config = VitsConfig(flow_size=2, upsample_initial_channel=8,
                            upsample_rates=[2, 2], upsample_kernel_sizes=[4, 4],
                            resblock_kernel_sizes=[3, 5],
                            resblock_dilation_sizes=[[1, 3], [1, 3]])
    else:
        config = VitsConfig.from_pretrained(args.model, local_files_only=True)
    decoder = VitsHifiGan(config).eval()
    if not tiny:
        state = load_file(Path(args.model) / "model.safetensors")
        decoder.load_state_dict({k.removeprefix("decoder."): v for k, v in state.items() if k.startswith("decoder.")}, strict=True)
    latent = torch.randn((1, config.flow_size, args.frames), generator=torch.Generator().manual_seed(41)) * .3
    with torch.inference_mode():
        waveform = decoder(latent)
    tensors = {"latent": latent[0].contiguous(), "waveform": waveform[0, 0].contiguous()}
    if tiny:
        tensors.update({"decoder." + k: v.contiguous() for k, v in decoder.state_dict().items()})
    metadata = {
        "reference": "transformers.VitsHifiGan",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "seed": "41",
        "frames": str(args.frames),
        "mode": "tiny" if tiny else "checkpoint",
        "config": json.dumps(config.to_dict(), sort_keys=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata=metadata)
    print(f"{args.output}: {config.flow_size}x{args.frames} latent -> {waveform.numel()} samples")


if __name__ == "__main__":
    main()
