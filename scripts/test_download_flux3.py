"""Offline checks for gated downloads and FLUX-specific component selection."""

import io
import json
import struct
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import download_flux3 as flux
import download_qwen_image21 as common


class FluxDownloadTests(unittest.TestCase):
    def test_text_tail_and_final_norm_are_not_used(self):
        self.assertTrue(flux.needed("text_encoder", "model.language_model.layers.31.mlp.up_proj.weight"))
        for name in ("model.language_model.layers.32.mlp.up_proj.weight",
                     "model.language_model.layers.35.input_layernorm.weight",
                     "model.language_model.norm.weight", "model.visual.patch_embed.proj.weight", "lm_head.weight"):
            self.assertFalse(flux.needed("text_encoder", name))
        self.assertTrue(flux.needed("text_encoder", "model.language_model.embed_tokens.weight"))
        self.assertTrue(flux.needed("vae", "bn.running_var"))
        self.assertFalse(flux.needed("vae", "bn.num_batches_tracked"))
        self.assertFalse(flux.needed("vae", "encoder.conv_in.weight"))
        self.assertFalse(flux.needed("transformer", "content_mode_blocks.image_cond.0.q_proj.weight"))
        self.assertTrue(flux.needed("transformer", "ref_time_in.in_layer.weight"))

    def test_missing_credentials_does_not_create_destination(self):
        with tempfile.TemporaryDirectory() as root, patch.object(flux, "get_token", return_value=None), patch.object(flux, "fetch") as fetch:
            dest = Path(root) / "model"
            with self.assertRaisesRegex(ValueError, "hf auth login"):
                flux.download(dest, 1)
            self.assertFalse(dest.exists())
            fetch.assert_not_called()

    def test_token_is_not_forwarded_to_redirects_or_cached_cdn(self):
        common._redirects.clear()
        requests = []

        class Response(io.BytesIO):
            status = 206
            def __init__(self, start):
                super().__init__(b"data")
                self.url = "https://cdn.example.invalid/object"
                self.headers = {"Content-Range": f"bytes {start}-{start+3}/100"}

        def urlopen(request, **kwargs):
            requests.append(request)
            start = int(request.get_header("Range").split("=")[1].split("-")[0])
            if len(requests) == 1:
                redirected = urllib.request.HTTPRedirectHandler().redirect_request(
                    request, None, 302, "Found", {}, "https://cdn.example.invalid/object")
                self.assertIsNone(redirected.get_header("Authorization"))
            return Response(start)

        with patch.object(common.urllib.request, "urlopen", side_effect=urlopen):
            common.fetch("https://huggingface.co/example/model", 0, 4, token="test-only-token")
            common.fetch("https://huggingface.co/example/model", 4, 4, token="test-only-token")
            common.fetch("http://huggingface.co/example/model", 0, 4, token="test-only-token")
        self.assertEqual(requests[0].get_header("Authorization"), "Bearer test-only-token")
        self.assertIsNone(requests[1].get_header("Authorization"))
        self.assertIsNone(requests[2].get_header("Authorization"))

    def test_shared_importer_filters_training_buffers_and_preserves_provenance(self):
        values = struct.pack("<128f", *[1.0] * 128)
        header = json.dumps({
            "bn.num_batches_tracked": {"dtype": "I64", "shape": [], "data_offsets": [0, 8]},
            "bn.running_var": {"dtype": "F32", "shape": [128], "data_offsets": [8, 520]},
        }).encode()
        blob = struct.pack("<Q", len(header)) + header + bytes(8) + values
        ranges = []

        def fetch(_, start, length):
            ranges.append((start, length))
            return blob[start:start+length]

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ae.gguf"
            names = common.import_shard("fixture", path, "vae", "pinned", 1,
                include=lambda name: flux.needed("vae", name),
                architecture="flux3_image_component", fetcher=fetch)
            self.assertEqual(names, ["bn.running_var"])
            self.assertIn(b"flux3_image_component", path.read_bytes())
            self.assertIn(b"pinned", path.read_bytes())
            self.assertTrue(path.read_bytes().endswith(values))
            self.assertNotIn((8 + len(header), 8), ranges)


if __name__ == "__main__":
    unittest.main()
