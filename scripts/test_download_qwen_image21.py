"""Offline downloader integrity, ordering and Q8_0 block tests."""

import hashlib
import importlib.util
import io
import json
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "download_qwen_image21", Path(__file__).with_name("download_qwen_image21.py")
)
DOWNLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOAD)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        DOWNLOAD._redirects.clear()

    def test_q8_expected_blocks(self):
        # Go QuantizeRowQ8_0: f16(amax/127), then half-away-from-zero
        # rounding using F32 products. In particular nextafter(.5,0) is 0.
        values = np.zeros(64, dtype="<f4")
        values[:6] = [
            -127,
            np.nextafter(np.float32(0.5), np.float32(0)),
            0.5,
            -np.nextafter(np.float32(0.5), np.float32(0)),
            -0.5,
            127,
        ]
        expected = (
            struct.pack("<H", 0x3C00)
            + np.array([-127, 0, 1, 0, -1, 127] + [0] * 26, dtype=np.int8).tobytes()
            + bytes(34)
        )
        self.assertEqual(DOWNLOAD.q8_blocks(values.tobytes(), "F32"), expected)

        # Storage conversion rounds the near-half values to exactly .5.
        expected = (
            struct.pack("<H", 0x3C00)
            + np.array([-127, 1, 1, -1, -1, 127] + [0] * 26, dtype=np.int8).tobytes()
            + bytes(34)
        )
        self.assertEqual(
            DOWNLOAD.q8_blocks(values.astype("<f2").tobytes(), "F16"), expected
        )
        exact = values.copy()
        exact[1] = 0.5
        exact[3] = -0.5
        raw = (exact.view("<u4") >> 16).astype("<u2").tobytes()
        self.assertEqual(DOWNLOAD.q8_blocks(raw, "BF16"), expected)

        values[0] = np.inf
        with self.assertRaises(ValueError):
            DOWNLOAD.q8_blocks(values.tobytes(), "F32")
        with self.assertRaises(ValueError):
            DOWNLOAD.q8_blocks(bytes(31 * 4), "F32")

    def test_prefetch_bounded_and_ordered(self):
        lock = threading.Lock()
        active = peak = 0

        def fetch(_, start, length):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.02 if start == 0 else 0.003)
                return bytes([start]) * length
            finally:
                with lock:
                    active -= 1

        with patch.object(DOWNLOAD, "CHUNK", 3), patch.object(DOWNLOAD, "fetch", fetch):
            got = list(DOWNLOAD.range_chunks("local", 0, 19))
        self.assertEqual(got, [bytes([i]) * min(3, 19 - i) for i in range(0, 19, 3)])
        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, 4)

    def test_prefetch_worker_limit(self):
        for workers in (0, 17):
            with self.assertRaises(ValueError):
                list(DOWNLOAD.range_chunks("local", 0, 3, workers))
        for workers in (1, 8, 16):
            with (
                self.subTest(workers=workers),
                patch.object(DOWNLOAD, "CHUNK", 3),
                patch.object(DOWNLOAD, "fetch", return_value=b"abc"),
            ):
                self.assertEqual(
                    list(DOWNLOAD.range_chunks("local", 0, 9, workers)), [b"abc"] * 3
                )

    def test_bf16_resume_rounding_is_unchanged(self):
        # The old F32 floor(abs(x)+.5) differs only immediately below .5.
        # Check the adjacent BF16 values around that boundary for every
        # possible positive BF16 block maximum with a finite inverse scale.
        maxima = (np.arange(1, 0x7F80, dtype="<u4") << 16).view("<f4")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            inverse = np.float32(1) / (maxima / np.float32(127))
        keep = np.isfinite(inverse)
        maxima = maxima[keep]
        inverse = inverse[keep]
        nearest = (np.float32(0.5) / inverse).view("<u4") >> 16
        for delta in range(-2, 3):
            bits = np.maximum(0, nearest.astype(np.int64) + delta).astype("<u4")
            values = (bits << 16).view("<f4")
            scaled = values * inverse
            old = np.floor(scaled + np.float32(0.5))
            precise = np.floor(scaled)
            precise += scaled - precise >= np.float32(0.5)
            valid = np.isfinite(values) & (values <= maxima)
            np.testing.assert_array_equal(old[valid], precise[valid])

    def test_resume_keeps_only_complete_tensors(self):
        header = b"GGUF header" + bytes(21)
        plan = [("a", {}, 0, 34, True), ("b", {}, 64, 34, True)]
        complete = header + bytes(range(34))
        for suffix in (bytes(30) + bytes(12), bytes(7), b""):
            with (
                self.subTest(suffix=len(suffix)),
                tempfile.TemporaryFile("w+b") as out,
            ):
                out.write(complete + suffix)
                count, position, digest = DOWNLOAD.checked_resume(out, header, plan)
                self.assertEqual((count, position), (1, 34))
                self.assertEqual(
                    digest.hexdigest(), hashlib.sha256(complete).hexdigest()
                )
                out.seek(0)
                self.assertEqual(out.read(), complete)

        with tempfile.TemporaryFile("w+b") as out:
            out.write(header[:5])
            count, position, digest = DOWNLOAD.checked_resume(out, header, plan)
            self.assertEqual((count, position), (0, 0))
            self.assertEqual(digest.hexdigest(), hashlib.sha256(header).hexdigest())

        for bad in (b"wrong header", header + bytes(99)):
            with tempfile.TemporaryFile("w+b") as out:
                out.write(bad)
                with self.assertRaises(ValueError):
                    DOWNLOAD.checked_resume(out, header, plan)
                out.seek(0)
                self.assertEqual(out.read(), bad)

    def test_completed_checksum_is_required_and_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "model.gguf"
            header = b"GGUF header"
            data = header + bytes(range(32))
            p.write_bytes(data)
            with self.assertRaises(ValueError):
                DOWNLOAD.verify_existing(p, header, len(data))

            p.with_suffix(".gguf.sha256").write_text(
                hashlib.sha256(data).hexdigest() + "\n"
            )
            DOWNLOAD.verify_existing(p, header, len(data))
            p.write_bytes(data[:-1] + b"x")
            with self.assertRaises(ValueError):
                DOWNLOAD.verify_existing(p, header, len(data))

    def test_import_resumes_without_redownloading_complete_tensor(self):
        bias = np.array([0.25, -0.5], dtype="<f4").tobytes()
        matrix = np.arange(-32, 32, dtype="<f4").reshape(2, 32).tobytes()
        index = {
            "a.bias": {"shape": [2], "dtype": "F32", "data_offsets": [0, 8]},
            "b.weight": {"shape": [2, 32], "dtype": "F32", "data_offsets": [8, 264]},
        }
        header = json.dumps(index).encode()
        blob = struct.pack("<Q", len(header)) + header + bias + matrix
        offset = 8 + len(header)
        calls = []
        fail = True

        def fetch(_, start, length):
            calls.append((start, length))
            if fail and start == offset + 128:
                raise OSError("interrupted transfer")
            return blob[start : start + length]

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(DOWNLOAD, "CHUNK", 128),
            patch.object(DOWNLOAD, "fetch", fetch),
        ):
            destination = Path(directory) / "model.gguf"
            with self.assertRaises(OSError):
                DOWNLOAD.import_shard("fixture", destination, "transformer", "revision")
            self.assertFalse(destination.exists())

            fail = False
            calls.clear()
            self.assertEqual(
                DOWNLOAD.import_shard(
                    "fixture", destination, "transformer", "revision"
                ),
                ["a.bias", "b.weight"],
            )
            self.assertNotIn((offset, len(bias)), calls)
            self.assertFalse(destination.with_suffix(".gguf.part").exists())
            self.assertTrue(
                destination.read_bytes().endswith(DOWNLOAD.q8_blocks(matrix, "F32"))
            )

            calls.clear()
            DOWNLOAD.import_shard("fixture", destination, "transformer", "revision")
            self.assertEqual(calls, [(0, 8), (8, len(header))])

    def test_prefetch_crosses_tensor_boundaries_and_skips_gaps(self):
        plan = [
            ("a", {"data_offsets": [0, 2]}),
            ("b", {"data_offsets": [2, 8]}),
            ("c", {"data_offsets": [11, 13]}),
        ]
        spans = list(DOWNLOAD.tensor_spans(plan))
        self.assertEqual(spans, [plan[:2], plan[2:]])
        requests = []

        def fetch(_, start, length):
            requests.append((start, length))
            return bytes(range(start, start + length))

        with patch.object(DOWNLOAD, "CHUNK", 3), patch.object(DOWNLOAD, "fetch", fetch):
            reader = DOWNLOAD.RangeReader(DOWNLOAD.range_chunks("local", 0, 8))
            self.assertEqual(bytes(reader.read(2)), bytes(range(2)))
            self.assertEqual(b"".join(reader.take(6)), bytes(range(2, 8)))
        self.assertEqual(sorted(requests), [(0, 3), (3, 3), (6, 2)])

    def test_cdn_redirect_reuse_and_expiry(self):
        calls = []

        class Response(io.BytesIO):
            status = 206

            def __init__(self, start):
                super().__init__(b"abcd")
                self.url = "https://cdn.example.invalid/object?signature=fixture"
                self.headers = {"Content-Range": f"bytes {start}-{start + 3}/100"}

        def open_range(request, **kwargs):
            calls.append(request.full_url)
            if len(calls) == 2:
                raise OSError("expired signature")
            start = int(request.get_header("Range").split("=")[1].split("-")[0])
            return Response(start)

        origin = "https://hub.example.invalid/model"
        with (
            patch.object(DOWNLOAD.urllib.request, "urlopen", side_effect=open_range),
            patch.object(DOWNLOAD.time, "sleep"),
        ):
            DOWNLOAD.fetch(origin, 8, 4)
            DOWNLOAD.fetch(origin, 12, 4)
        self.assertEqual(
            calls,
            [
                origin + "?range=8-11",
                "https://cdn.example.invalid/object?signature=fixture",
                origin + "?range=12-15",
            ],
        )

    def test_fetch_rejects_malformed_range(self):
        class Response(io.BytesIO):
            def __init__(self, status, content_range, body):
                super().__init__(body)
                self.status = status
                self.headers = {"Content-Range": content_range}

        for status, header, body in (
            (200, "bytes 8-11/100", b"abcd"),
            (206, "bytes 8-12/100", b"abcd"),
            (206, "bytes 8-11/*", b"abcd"),
            (206, "bytes 8-11/11", b"abcd"),
            (206, "bytes 8-11/100", b"abc"),
            (206, "bytes 8-11/100", b"abcde"),
        ):
            with (
                self.subTest(status=status, header=header, body=body),
                patch.object(
                    DOWNLOAD.urllib.request,
                    "urlopen",
                    side_effect=lambda *a, status=status, header=header, body=body, **kw: (
                        Response(status, header, body)
                    ),
                ),
                patch.object(DOWNLOAD.time, "sleep"),
                self.assertRaises(ValueError),
            ):
                DOWNLOAD.fetch("https://example.invalid/model", 8, 4)

        with patch.object(
            DOWNLOAD.urllib.request,
            "urlopen",
            side_effect=lambda *a, **kw: Response(206, "bytes 8-11/100", b"abcd"),
        ):
            self.assertEqual(
                DOWNLOAD.fetch("https://example.invalid/model", 8, 4), b"abcd"
            )


if __name__ == "__main__":
    unittest.main()
