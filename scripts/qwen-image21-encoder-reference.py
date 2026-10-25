#!/usr/bin/env python3
"""Build seeded Qwen3-VL text fixtures; Python is not used by Go inference.

Reference: transformers 4.57.1 Qwen3VLTextModel, PyTorch 2.8.0, tokenizers 0.22.2.
The weights are random and contain no downloaded model parameters.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import Qwen3VLTextConfig, Qwen3VLTextModel

PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
PREFIX = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"


def byte_map():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(161, 173))
        + list(range(174, 256))
    )
    cs = list(bs)
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + sum(c >= 256 for c in cs))
    return dict(zip(bs, map(chr, cs)))


def tiny_tokenizer():
    vocab = {c: b for b, c in byte_map().items()}
    vocab.update({"Ġh": 256, "Ġhi": 257})
    return {
        "version": "1.0",
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": PATTERN},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": False,
                    "use_regex": False,
                },
            ],
        },
        "added_tokens": [
            {
                "id": 258 + i,
                "content": s,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for i, s in enumerate(("<|im_start|>", "<|im_end|>", "<|endoftext|>"))
        ],
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [["Ġ", "h"], ["Ġh", "i"]],
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="testdata/qwen_image21_encoder")
    parser.add_argument(
        "--official-tokenizer", help="optional downloaded processor/tokenizer.json"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(24581)
    cfg = Qwen3VLTextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=300,
        max_position_embeddings=512,
        rope_theta=5000000,
        rope_scaling={
            "rope_type": "default",
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
        },
        rms_norm_eps=1e-6,
        attention_bias=False,
    )
    cfg._attn_implementation = "eager"
    model = Qwen3VLTextModel(cfg).float().eval()
    # Nonuniform norms expose omitted/misplaced QK norms; final norm must not
    # appear in the conditioning features, even with a conspicuous weight.
    with torch.no_grad():
        for name, value in model.named_parameters():
            if value.ndim == 1:
                value.copy_(torch.linspace(0.7, 1.3, value.numel()))
        model.norm.weight.fill_(2.5)
    tokenizer_config = tiny_tokenizer()
    tokenizer = Tokenizer.from_str(json.dumps(tokenizer_config))
    prompt = " hi  123 café e\u0301!\nA blue cube."
    encoded = tokenizer.encode(
        PREFIX + "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    ).ids
    drop = len(tokenizer.encode(PREFIX).ids)
    hook = model.norm.register_forward_hook(lambda module, inputs, output: inputs[0])
    with torch.no_grad():
        hidden = (
            model(torch.tensor([encoded]), use_cache=False)
            .last_hidden_state[0, drop:]
            .contiguous()
        )
    hook.remove()
    values = {
        "model.language_model." + k: v.detach().contiguous()
        for k, v in model.state_dict().items()
    }
    values["reference.hidden"] = hidden
    save_file(
        values,
        args.output + ".safetensors",
        metadata={
            "source": "transformers 4.57.1 Qwen3VLTextModel; torch 2.8.0; seeded random weights"
        },
    )
    Path(args.output + ".json").write_text(
        json.dumps(
            {
                "config": {"model_type": "qwen3_vl", "text_config": cfg.to_dict()},
                "tokenizer": tokenizer_config,
                "prompt": prompt,
                "tokens": encoded,
                "drop": drop,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    if args.official_tokenizer:
        actual = Tokenizer.from_file(args.official_tokenizer)
        inverse = {v: k for k, v in byte_map().items()}
        samples = [
            "A blue cube.",
            "",
            "  hello   world",
            "12345 Ⅷ² 你好１２",
            "WE'RE can't I'm",
            "café e\u0301",
            "\u0301a a\u0301\u0302 ऑ",
            "\t hi\r\n  there\n\t",
            "🙂 !?\nnext",
            PREFIX,
            PREFIX
            + "<|im_start|>user\nA blue cube.<|im_end|>\n<|im_start|>assistant\n",
        ]
        rows = []
        for s in samples:
            normalized = actual.normalizer.normalize_str(s)
            pieces = [
                bytes(inverse[c] for c in p).decode("utf-8")
                for p, _ in actual.pre_tokenizer.pre_tokenize_str(normalized)
            ]
            rows.append({"text": s, "pieces": pieces, "tokens": actual.encode(s).ids})
        Path("testdata/qwen_image21_tokenizer_reference.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
