#!/usr/bin/env python3
"""Generate MMS/VITS numerical references; never used by Go inference.

Requires Python >=3.11, torch==2.8.0, transformers==4.57.1 and safetensors==0.6.2.
The input is an already downloaded facebook/mms-tts-deu directory. This script
does not fetch models. Check the checkpoint's CC-BY-NC-4.0 license separately.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import VitsModel, VitsTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    model = VitsModel.from_pretrained(args.model, local_files_only=True).eval()
    tokenizer = VitsTokenizer.from_pretrained(args.model, local_files_only=True)
    text = "Hallo Welt."
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    values = {"ids": ids.float()}

    def record(name, tensor):
        values[name] = tensor.detach().cpu().float().contiguous().clone()

    def encoded(_module, _inputs, output):
        record("hidden", output.last_hidden_state.transpose(1, 2))
        record("means", output.prior_means.transpose(1, 2))
        record("logs", output.prior_log_variances.transpose(1, 2))

    model.text_encoder.register_forward_hook(encoded)
    model.duration_predictor.register_forward_hook(
        lambda _module, _inputs, output: record("log_duration", output)
    )
    model.flow.register_forward_pre_hook(
        lambda _module, inputs: record("prior_latents", inputs[0])
    )
    model.flow.register_forward_hook(
        lambda _module, _inputs, output: record("latents", output)
    )

    # Record the actual Gaussian draws rather than replacing stochastic
    # inference with zero noise or expecting Go and PyTorch RNGs to match.
    original_randn, original_like = torch.randn, torch.randn_like

    def randn(*arguments, **kwargs):
        result = original_randn(*arguments, **kwargs)
        record("duration_noise", result * model.noise_scale_duration)
        return result

    def randn_like(*arguments, **kwargs):
        result = original_like(*arguments, **kwargs)
        record("latent_noise", result)
        return result

    try:
        torch.randn, torch.randn_like = randn, randn_like
        with torch.inference_mode():
            result = model(ids)
    finally:
        torch.randn, torch.randn_like = original_randn, original_like
    record("waveform", result.waveform)
    with (args.model / "model.safetensors").open("rb") as stream:
        model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "text": text,
        "seed": str(args.seed),
        "model_sha256": model_hash,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(values, str(args.output), metadata=metadata)
    print(json.dumps({**metadata, "samples": result.waveform.numel(),
                      "sample_rate": model.config.sampling_rate,
                      "reference": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
