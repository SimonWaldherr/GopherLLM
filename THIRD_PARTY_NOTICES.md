# Qwen-Image-2.1 reference implementations

The native Go transformer, flow scheduler and image VAE were adapted from the
Diffusers Qwen-Image-2.1 implementation. They are modified Go implementations,
not the original Python code.

- Transformer: Copyright 2026 Qwen-Image Team, The HuggingFace Team.
- VAE: Copyright 2026 The Qwen Team and The HuggingFace Team.
- Flow Matching scheduler: Copyright 2025 Stability AI, Katherine Crowson and
  The HuggingFace Team.

These references are licensed under the Apache License, Version 2.0; a copy is
included in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt). Source references:

- [Qwen-Image-2.1 introduction commit](https://github.com/huggingface/diffusers/commit/6256aa7666cedd47443adc8f82da9a10e110b09c)
- [VAE reference](https://github.com/huggingface/diffusers/blob/aaf57d03a3edfbd9de457d2872417a92f149a91d/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py)

Seeded test fixtures use randomly initialized reference networks and contain
no pretrained Qwen weights. Downloaded checkpoints retain the upstream Qwen
Research License; the code license does not replace model usage terms.
