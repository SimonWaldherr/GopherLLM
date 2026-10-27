#!/usr/bin/env python3
"""Download pinned Qwen Image 2.1 components as mmap-friendly GGUF shards.

Requires Python 3.11+ and NumPy for offline Q8_0 conversion; Go inference does
not use Python.
HTTP ranges avoid retaining a second 33 GB checkpoint while converting it.
The component tensors retain their upstream names. Only 2D matrices are
quantized; norms and VAE convolutions keep their original floating-point type.
"""

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import numpy as np

REVISIONS = {
    "base": ("Qwen/Qwen-Image-2.1", "d26bb61231c349cf6b7896fa83353113880e1ba3"),
    "turbo": ("Qwen/Qwen-Image-2.1-Turbo", "fc13ba1986fcb905854dc8de2f52eee395f49b5b"),
}
DTYPES = {"F32": (0, 4), "F16": (1, 2), "BF16": (30, 2)}
CHUNK = 16 << 20
_redirects = {}
_redirect_lock = threading.Lock()


def fetch(url, start=None, length=None, *, token=None):
    """Require exact ranges; never accidentally consume a full large file."""
    headers = {"User-Agent": "GopherLLM-QwenImage21"}
    if start is not None:
        if (
            not isinstance(start, int)
            or not isinstance(length, int)
            or start < 0
            or length <= 0
        ):
            raise ValueError("invalid HTTP range")
        headers["Range"] = f"bytes={start}-{start + length - 1}"
    origin = url
    for attempt in range(4):
        with _redirect_lock:
            resolved = _redirects.get(origin) if start is not None else None
        request_url = resolved or origin
        if start is not None and resolved is None:
            request_url += (
                "&" if "?" in request_url else "?"
            ) + f"range={start}-{start + length - 1}"
        try:
            request = urllib.request.Request(request_url, headers=headers)
            parsed = urllib.parse.urlsplit(request_url)
            if token and parsed.scheme == "https" and parsed.hostname == "huggingface.co":
                # Credentials belong only to the Hub origin. urllib omits
                # unredirected headers when following signed CDN redirects.
                request.add_unredirected_header("Authorization", "Bearer " + token)
            with urllib.request.urlopen(request, timeout=90) as response:
                if start is not None:
                    bounds = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)",
                        response.headers.get("Content-Range", ""),
                    )
                    if (
                        response.status != 206
                        or bounds is None
                        or tuple(map(int, bounds.groups()[:2]))
                        != (start, start + length - 1)
                        or int(bounds.group(3)) < start + length
                    ):
                        raise ValueError(
                            "server did not honor the requested byte range"
                        )
                data = response.read((length if length is not None else 16 << 20) + 1)
                if length is not None and len(data) != length:
                    raise ValueError("incomplete or oversized HTTP range")
                if length is None and len(data) > 16 << 20:
                    raise ValueError("metadata exceeds 16 MiB")
                final_url = getattr(response, "url", request_url)
                if (
                    start is not None
                    and urllib.parse.urlsplit(final_url).netloc
                    != urllib.parse.urlsplit(origin).netloc
                ):
                    with _redirect_lock:
                        _redirects[origin] = final_url
                return data
        except (OSError, ValueError):
            # CDN signatures expire. Retry through the pinned Hub URL instead
            # of persisting or logging a signed redirect.
            if resolved is not None:
                with _redirect_lock:
                    if _redirects.get(origin) == resolved:
                        _redirects.pop(origin, None)
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def text(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def q8_blocks(raw, dtype):
    if dtype == "BF16":
        values = (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
    else:
        values = np.frombuffer(raw, dtype="<f4" if dtype == "F32" else "<f2").astype(
            np.float32
        )
    values = values.reshape(-1, 32)
    if not np.isfinite(values).all():
        raise ValueError("non-finite source weights")
    scales = np.abs(values).max(axis=1) / np.float32(127)
    inverse = np.zeros_like(scales)
    np.divide(np.float32(1), scales, out=inverse, where=scales != 0)
    scaled = values * inverse[:, None]
    # Adding 0.5 in F32 can round a value just below a half-integer up to
    # the next integer. Compare the fractional part instead (Go math.Round).
    magnitude = np.abs(scaled)
    rounded = np.floor(magnitude)
    rounded += magnitude - rounded >= np.float32(0.5)
    np.copysign(rounded, scaled, out=rounded)
    packed = np.empty((len(scales), 34), dtype=np.uint8)
    packed[:, :2] = scales.astype("<f2").view(np.uint8).reshape(-1, 2)
    packed[:, 2:] = np.clip(rounded, -127, 127).astype(np.int8).view(np.uint8)
    return packed.tobytes()


def needed(component, name):
    if component != "text_encoder":
        return True
    return (
        name.startswith("model.language_model.layers.")
        or name == "model.language_model.embed_tokens.weight"
    )


def range_chunks(url, begin, end, workers=4, *, fetcher=None):
    """Prefetch a bounded number of ranges and consume them in file order."""
    if begin < 0 or end < begin or CHUNK <= 0 or not 1 <= workers <= 16:
        raise ValueError("invalid source span")
    fetcher = fetcher or fetch
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        position = begin
        while position < end or pending:
            while position < end and len(pending) < workers:
                count = min(CHUNK, end - position)
                pending.append(pool.submit(fetcher, url, position, count))
                position += count
            yield pending.popleft().result()


def tensor_spans(plan):
    """Keep prefetch running across contiguous tensor boundaries."""
    span = []
    for item in plan:
        if span and span[-1][1]["data_offsets"][1] != item[1]["data_offsets"][0]:
            yield span
            span = []
        span.append(item)
    if span:
        yield span


class RangeReader:
    """Split prefetched source ranges without retaining more than one range."""

    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.buffer = memoryview(b"")

    def read(self, count):
        if not self.buffer:
            self.buffer = memoryview(next(self.chunks))
        if len(self.buffer) >= count:
            result, self.buffer = self.buffer[:count], self.buffer[count:]
            return result
        result = bytearray(count)
        position = 0
        while position < count:
            if not self.buffer:
                self.buffer = memoryview(next(self.chunks))
            take = min(count - position, len(self.buffer))
            result[position : position + take] = self.buffer[:take]
            self.buffer = self.buffer[take:]
            position += take
        return result

    def take(self, count):
        while count:
            size = min(CHUNK, count)
            yield self.read(size)
            count -= size


def gguf_header(plan, url, revision, architecture="qwen_image21_component"):
    metadata = {
        "general.architecture": architecture,
        "general.source.url": url,
        "general.source.revision": revision,
    }
    header = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(plan), len(metadata)))
    for key, value in metadata.items():
        header += text(key) + struct.pack("<I", 8) + text(value)
    for name, tensor, begin, _, quantize in plan:
        shape = tensor["shape"]
        header += text(name) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *reversed(shape))
        header += struct.pack(
            "<IQ", 8 if quantize else DTYPES[tensor["dtype"]][0], begin
        )
    header += b"\0" * (-len(header) % 32)
    return bytes(header)


def checked_resume(out, header, plan):
    """Validate provenance/layout before discarding an unfinished tensor."""
    size = out.seek(0, 2)
    out.seek(0)
    if out.read(min(size, len(header))) != header[: min(size, len(header))]:
        raise ValueError("partial GGUF header differs from this checkpoint")
    expected_size = len(header) + plan[-1][2] + plan[-1][3]
    if size > expected_size:
        raise ValueError("partial GGUF exceeds expected size")
    count, position = 0, 0
    for _, _, begin, length, _ in plan:
        if len(header) + begin + length > size:
            break
        count += 1
        position = begin + length
    if size < len(header):
        out.seek(0)
        out.write(header)
    out.truncate(len(header) + position)
    out.seek(0)
    digest = hashlib.file_digest(out, "sha256")
    out.seek(len(header) + position)
    return count, position, digest


def verify_existing(destination, header, expected_size):
    if destination.stat().st_size != expected_size:
        raise ValueError(f"existing shard has the wrong size: {destination}")
    checksum = destination.with_suffix(".gguf.sha256")
    if not checksum.is_file():
        raise ValueError(f"missing checksum for existing shard: {destination}")
    expected = checksum.read_text().strip()
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"invalid checksum for existing shard: {destination}")
    with destination.open("rb") as file:
        if file.read(len(header)) != header:
            raise ValueError(f"existing shard header differs: {destination}")
        file.seek(0)
        actual = hashlib.file_digest(file, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"existing shard checksum mismatch: {destination}")


def import_shard(url, destination, component, revision, workers=4, *,
                 include=None, architecture="qwen_image21_component", fetcher=None):
    fetcher = fetcher or fetch
    include = include or (lambda name: needed(component, name))
    header_size = struct.unpack("<Q", fetcher(url, 0, 8))[0]
    if not 0 < header_size <= 16 << 20:
        raise ValueError("invalid safetensors header length")
    source = json.loads(fetcher(url, 8, header_size))
    plan, offset = [], 0
    for name, tensor in sorted(source.items()):
        if name == "__metadata__" or not include(name):
            continue
        shape, dtype = tensor["shape"], tensor["dtype"]
        if (
            not shape
            or len(shape) > 5
            or any(type(n) is not int or not 0 < n <= 1 << 24 for n in shape)
            or dtype not in DTYPES
        ):
            raise ValueError(f"unsupported tensor {name}")
        elements = math.prod(shape)
        begin, end = tensor["data_offsets"]
        if (
            type(begin) is not int
            or type(end) is not int
            or begin < 0
            or end - begin != elements * DTYPES[dtype][1]
        ):
            raise ValueError(f"invalid storage for {name}")
        quantize = component != "vae" and len(shape) == 2 and shape[-1] % 32 == 0
        size = elements // 32 * 34 if quantize else end - begin
        offset = (offset + 31) // 32 * 32
        plan.append((name, tensor, offset, size, quantize))
        offset += size
    if not plan:
        return []
    names = [item[0] for item in plan]
    header = gguf_header(plan, url, revision, architecture)
    if destination.exists():
        verify_existing(destination, header, len(header) + offset)
        print(f"verified {destination.name}", flush=True)
        return names
    temporary = destination.with_suffix(".gguf.part")
    with temporary.open("r+b" if temporary.exists() else "w+b") as out:
        completed, position, digest = checked_resume(out, header, plan)
        remaining = offset - position
        if shutil.disk_usage(destination.parent).free < remaining + (512 << 20):
            raise OSError(
                f"need {remaining + (512 << 20)} free bytes for {destination}"
            )
        if completed:
            print(
                f"resume {destination.name}: {completed} tensors, {position / (1 << 30):.2f} GiB",
                flush=True,
            )
        for span in tensor_spans(plan[completed:]):
            first = 8 + header_size + span[0][1]["data_offsets"][0]
            last = 8 + header_size + span[-1][1]["data_offsets"][1]
            with closing(range_chunks(url, first, last, workers, fetcher=fetcher)) as ranges:
                reader = RangeReader(ranges)
                for name, tensor, begin, size, quantize in span:
                    padding = b"\0" * (begin - position)
                    out.write(padding)
                    digest.update(padding)
                    source_begin, source_end = tensor["data_offsets"]
                    written = 0
                    for raw in reader.take(source_end - source_begin):
                        data = q8_blocks(raw, tensor["dtype"]) if quantize else raw
                        out.write(data)
                        digest.update(data)
                        written += len(data)
                    if written != size:
                        raise ValueError(f"converted tensor has the wrong size: {name}")
                    position = begin + size
                    if name.endswith(("o_proj.weight", "linear2.weight")):
                        print(
                            f"  {name}: {position / (1 << 30):.2f} GiB written",
                            flush=True,
                        )
    # Publish the checksum first, so a completed destination is never reused
    # without an integrity record after interruption between the two renames.
    checksum = destination.with_suffix(".gguf.sha256")
    checksum_temp = checksum.with_name(checksum.name + ".part")
    checksum_temp.write_text(digest.hexdigest() + "\n")
    checksum_temp.replace(checksum)
    temporary.replace(destination)
    print(
        f"saved {destination.name}: {destination.stat().st_size / (1 << 30):.2f} GiB",
        flush=True,
    )
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--variant", choices=REVISIONS, default="turbo")
    parser.add_argument(
        "--workers",
        type=int,
        choices=range(1, 17),
        default=4,
        metavar="1..16",
        help="parallel byte-range requests (default: 4)",
    )
    args = parser.parse_args()
    repo, revision = REVISIONS[args.variant]
    args.directory.mkdir(parents=True, exist_ok=True)
    provenance = {
        "repository": repo,
        "revision": revision,
        "matrices": "Q8_0",
        "runtime": "native Go",
    }
    source_path = args.directory / "source.json"
    if source_path.exists() and json.loads(source_path.read_text()) != provenance:
        raise ValueError("destination belongs to a different checkpoint")
    source_path.write_text(json.dumps(provenance, indent=2) + "\n")
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    model = json.loads(
        fetch(
            f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
        )
    )
    files = [entry["rfilename"] for entry in model["siblings"]]
    metadata = [
        name
        for name in files
        if name in ("LICENSE", "README.md", "model_index.json")
        or any(
            name.startswith(c + "/")
            for c in ("processor", "scheduler", "text_encoder", "transformer", "vae")
        )
        and name.endswith((".json", ".jinja"))
        and ".safetensors.index." not in name
    ]
    for name in metadata:
        dest = args.directory / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(fetch(base + urllib.parse.quote(name)))
    for component in ("transformer", "text_encoder", "vae"):
        mapping = {}
        for name in sorted(
            n
            for n in files
            if n.startswith(component + "/") and n.endswith(".safetensors")
        ):
            destination = args.directory / Path(name).with_suffix(".gguf")
            destination.parent.mkdir(parents=True, exist_ok=True)
            for tensor in import_shard(
                base + urllib.parse.quote(name),
                destination,
                component,
                revision,
                args.workers,
            ):
                if tensor in mapping:
                    raise ValueError(f"duplicate tensor {tensor}")
                mapping[tensor] = destination.name
        if not mapping:
            raise ValueError(f"no weights found for {component}")
        index = args.directory / component / "weights.index.json"
        temp = index.with_suffix(".tmp")
        temp.write_text(json.dumps({"weight_map": mapping}, indent=2) + "\n")
        temp.replace(index)


if __name__ == "__main__":
    main()
