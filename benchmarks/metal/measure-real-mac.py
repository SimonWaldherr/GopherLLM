"""Run one opt-in decoder measurement; RSS is per process, VM counters global."""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return seconds


def validate_records(records, kind):
    """Reject partial runs and rates that cannot describe the emitted workload."""
    if kind not in ("ministral", "voxtral"):
        raise ValueError("unknown decoder kind")
    expected = 3 if kind == "voxtral" else 4
    phases = ["before_load", "loaded"]
    for _ in range(expected):
        phases.append("decode")
        if kind == "voxtral":
            phases.append("decoder_closed")
    if any(not isinstance(r, dict) for r in records) or [r.get("phase") for r in records] != phases:
        raise ValueError("incomplete or out-of-order decoder measurements; check the test binary and log")
    load_seconds = records[1].get("seconds")
    if type(load_seconds) not in (int, float) or not math.isfinite(load_seconds) or load_seconds <= 0:
        raise ValueError("load duration must be finite and positive")
    digests = set()
    repetition = 0
    for record in records:
        if record["phase"] == "decoder_closed":
            if type(record.get("repetition")) is not int or record["repetition"] != repetition - 1:
                raise ValueError("decoder close repetition does not match decode")
        if record["phase"] != "decode":
            continue
        seconds = record.get("seconds")
        rate = record.get("tokens_per_second")
        prefill = record.get("prefill_seconds")
        for value in (seconds, rate, prefill):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("decode timings and rates must be finite numbers")
        if type(record.get("tokens")) is not int or record["tokens"] != 64 or seconds <= 0 or rate <= 0 or prefill < 0:
            raise ValueError("unexpected decode length or duration")
        if type(record.get("repetition")) is not int or record["repetition"] != repetition:
            raise ValueError("decode repetitions must be consecutive, starting at zero")
        if not math.isclose(rate, 64 / seconds, rel_tol=1e-9):
            raise ValueError("decode rate does not match token count and duration")
        digest = record.get("token_digest")
        # TestRealMacDecode uses specABDigest's 128-bit FNV-1a token digest.
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{32}", digest):
            raise ValueError("missing or invalid token digest")
        digests.add(digest)
        repetition += 1
    if len(digests) != 1:
        raise ValueError("repeated decoder tokens differ")


def parse_vm_stat(output):
    page_size = re.search(r"page size of (\d+)", output)
    counters = {k: int(v) for k, v in re.findall(r"^([^:\n]+):\s+(\d+)\.", output, re.M)}
    if not page_size or int(page_size[1]) <= 0 or not {"Swapins", "Swapouts"} <= counters.keys():
        raise ValueError("vm_stat output lacks page size or swap counters")
    return {"page_size": int(page_size[1]), "counters": counters}


def command_output(command, deadline=None):
    timeout = 5
    if deadline is not None:
        timeout = min(timeout, deadline - time.monotonic())
        if timeout <= 0:
            raise TimeoutError("model measurement timed out")
    return subprocess.check_output(command, text=True, timeout=timeout)


def vm(deadline=None):
    snapshot = parse_vm_stat(command_output(["/usr/bin/vm_stat"], deadline))
    snapshot["swapusage"] = command_output(["/usr/sbin/sysctl", "vm.swapusage"], deadline).strip()
    return snapshot


def measure(binary, model, kind, copy, output, timeout):
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    log = output / "run.log"
    # Keep earlier evidence intact and prevent failed reruns from appearing valid.
    result = output / "run.json"
    if result.exists() or log.exists():
        raise FileExistsError("run.log or run.json already exists; use a fresh output directory")
    before = vm()
    started = time.time()
    clock_start = time.monotonic()
    deadline = clock_start + timeout
    samples = []
    env = dict(os.environ, GOMAXPROCS="8", GOPHERLLM_REAL_MAC_KIND=kind,
               GOPHERLLM_REAL_MAC_MODEL=str(model.resolve()), GOPHERLLM_REAL_MAC_COPY="1" if copy else "0")
    with log.open("x") as stream:
        process = subprocess.Popen(
            [str(binary.resolve()), "-test.run", "^TestRealMacDecode$", "-test.v", "-test.count=1"],
            cwd=root / "internal/engine", env=env, stdout=stream, stderr=subprocess.STDOUT,
        )
        try:
            while True:
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    process.returncode = os.waitstatus_to_exitcode(status)
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("model measurement timed out")
                try:
                    # ps reports KiB; Darwin wait4 reports maximum RSS in bytes.
                    rss = int(command_output(
                        ["/bin/ps", "-o", "rss=", "-p", str(process.pid)], deadline
                    ).strip()) * 1024
                except (subprocess.CalledProcessError, ValueError):
                    rss = None
                samples.append({"unix_time": time.time(), "rss_bytes": rss, "vm": vm(deadline)})
                time.sleep(min(0.5, max(0, deadline - time.monotonic())))
        except BaseException:
            if process.returncode is None:
                process.kill()
                process.wait()
            raise
    elapsed = time.monotonic() - clock_start
    after = vm()
    records = [json.loads(line.removeprefix("REAL_MAC_JSON ")) for line in log.read_text().splitlines()
               if line.startswith("REAL_MAC_JSON ")]
    if process.returncode == 0:
        validate_records(records, kind)
    out = {"kind": kind, "loader_copy": copy, "model": model.name, "mode": "run", "started": started,
           "elapsed_s": elapsed, "returncode": process.returncode, "before": before, "after": after,
           "records": records, "samples": samples, "maxrss_bytes": usage.ru_maxrss,
           "major_faults": usage.ru_majflt, "minor_faults": usage.ru_minflt}
    result.write_text(json.dumps(out, indent=2, allow_nan=False))
    print("run exit", process.returncode, "seconds", elapsed, "maxrssGiB", usage.ru_maxrss / 2 ** 30, flush=True)
    if process.returncode:
        print(log.read_text()[-2500:], flush=True)
    # Report the conventional 128 + signal exit status for a killed child.
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=["ministral", "voxtral"])
    parser.add_argument("--copy", action="store_true", help="Use the prior copying Voxtral loader as control")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=positive_seconds, default=300)
    args = parser.parse_args(argv)
    if args.copy and args.kind != "voxtral":
        parser.error("--copy only applies to Voxtral")
    if sys.platform != "darwin":
        parser.error("this sampler requires macOS (vm_stat and Darwin RSS units)")
    try:
        return measure(args.binary, args.model, args.kind, args.copy, args.output.resolve(), args.timeout)
    except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
        parser.exit(1, f"measurement failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
