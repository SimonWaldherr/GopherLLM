"""Sampler contracts, runnable without macOS, model weights or a GPU."""

import contextlib
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("measure_real_mac", Path(__file__).with_name("measure-real-mac.py"))
sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler)


def records(kind="ministral"):
    result = [{"phase": "before_load"}, {"phase": "loaded", "seconds": 1.0}]
    for rep in range(3 if kind == "voxtral" else 4):
        result.append({"phase": "decode", "repetition": rep, "tokens": 64, "seconds": 2.0,
                       "tokens_per_second": 32.0, "prefill_seconds": 0.0, "token_digest": "a" * 32})
        if kind == "voxtral":
            result.append({"phase": "decoder_closed", "repetition": rep})
    return result


class RecordTests(unittest.TestCase):
    def test_accepts_complete_runs(self):
        for kind in ("ministral", "voxtral"):
            with self.subTest(kind=kind):
                sampler.validate_records(records(kind), kind)

    def test_rejects_invalid_load_duration_and_decoder_kind(self):
        for seconds in (None, "1", True, 0, -1, math.nan, math.inf):
            with self.subTest(seconds=seconds):
                data = records()
                data[1]["seconds"] = seconds
                with self.assertRaisesRegex(ValueError, "load duration"):
                    sampler.validate_records(data, "ministral")
        with self.assertRaisesRegex(ValueError, "unknown decoder"):
            sampler.validate_records(records(), "other")

    def test_recorded_checkpoint_runs_remain_valid(self):
        data = json.loads(Path(__file__).with_name("real-mac-32gb-2026-10-30.json").read_text())
        for run in data["runs"]:
            with self.subTest(mode=run["mode"]):
                kind = "voxtral" if run["model"].startswith("Voxtral") else "ministral"
                sampler.validate_records(run["records"], kind)

    def test_rejects_partial_or_reordered_phases(self):
        valid = records()
        cases = [[], valid[1:], valid[:-1], valid + [valid[-1]], valid[::-1], [None]]
        voxtral = records("voxtral")
        cases.append(voxtral)
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    sampler.validate_records(data, "ministral")
        with self.assertRaises(ValueError):
            sampler.validate_records(voxtral[:-1], "voxtral")

    def test_rejects_invalid_decode_fields(self):
        cases = {
            "tokens": [None, 63, 65, 64.0],
            "seconds": [None, "2", True, 0, -1, math.nan, math.inf],
            "tokens_per_second": [None, 31.0, 0, -1, math.nan, math.inf],
            "prefill_seconds": [None, -1, math.nan, math.inf],
            "token_digest": [None, "", "a" * 31, "z" * 32, "b" * 32],
            "repetition": [None, 0, 4, True, 1.0],
        }
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    data = records()
                    data[3][field] = value
                    with self.assertRaises(ValueError):
                        sampler.validate_records(data, "ministral")
        data = records("voxtral")
        data[3]["repetition"] = 2
        with self.assertRaisesRegex(ValueError, "close repetition"):
            sampler.validate_records(data, "voxtral")
        data[3]["repetition"] = False
        with self.assertRaisesRegex(ValueError, "close repetition"):
            sampler.validate_records(data, "voxtral")

    def test_accepts_small_rate_roundoff(self):
        data = records()
        data[2]["tokens_per_second"] += 1e-10
        sampler.validate_records(data, "ministral")

    def test_vm_stat_page_sizes_and_counters(self):
        for page_size in (4096, 16384):
            with self.subTest(page_size=page_size):
                output = f"Mach Virtual Memory Statistics: (page size of {page_size} bytes)\n"
                output += "Pages free:                             17.\nSwapins: 23.\nSwapouts: 42.\n"
                self.assertEqual(sampler.parse_vm_stat(output), {
                    "page_size": page_size, "counters": {"Pages free": 17, "Swapins": 23, "Swapouts": 42}
                })
        for output in ("", "page size of 0\nSwapins: 1.\nSwapouts: 2.", "page size of 16384\nPages free: 0."):
            with self.subTest(output=output), self.assertRaises(ValueError):
                sampler.parse_vm_stat(output)

    def test_vm_commands_share_the_remaining_deadline(self):
        output = "page size of 16384\nSwapins: 1.\nSwapouts: 2."
        with patch.object(sampler.subprocess, "check_output", side_effect=[output, "swap usage"] ) as read:
            with patch.object(sampler.time, "monotonic", side_effect=[100, 100.03]):
                sampler.vm(deadline=100.05)
        self.assertAlmostEqual(read.call_args_list[0].kwargs["timeout"], 0.05)
        self.assertAlmostEqual(read.call_args_list[1].kwargs["timeout"], 0.02)

    def test_expired_deadline_does_not_launch_telemetry_command(self):
        with patch.object(sampler.subprocess, "check_output") as read:
            with patch.object(sampler.time, "monotonic", return_value=100):
                with self.assertRaises(TimeoutError):
                    sampler.command_output(["unused"], deadline=99)
        read.assert_not_called()

    def test_invalid_cli_arguments_do_not_start_measurement(self):
        base = ["--binary", "engine.test", "--model", "model.gguf", "--output", "unused", "--kind", "ministral"]
        for extra in (["--copy"], ["--timeout", "0"], ["--timeout", "-1"], ["--timeout", "nan"], ["--timeout", "inf"]):
            with self.subTest(extra=extra), patch.object(sampler, "measure") as measure:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    sampler.main(base + extra)
                self.assertEqual(error.exception.code, 2)
                measure.assert_not_called()


@unittest.skipUnless(hasattr(os, "wait4"), "requires POSIX child resource accounting")
class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / "fake engine.test"
        self.model = self.root / "model.gguf"
        self.output = self.root / "result"
        self.snapshot = {"page_size": 16384, "counters": {"Swapins": 0, "Swapouts": 0}, "swapusage": "test"}
        vm = patch.object(sampler, "vm", return_value=self.snapshot)
        vm.start()
        self.addCleanup(vm.stop)
        ps = patch.object(sampler.subprocess, "check_output", return_value="123")
        ps.start()
        self.addCleanup(ps.stop)
        self.children = []
        popen = subprocess.Popen

        def capture(*args, **kwargs):
            child = popen(*args, **kwargs)
            self.children.append(child)
            return child

        children = patch.object(sampler.subprocess, "Popen", side_effect=capture)
        children.start()
        self.addCleanup(children.stop)
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def script(self, body):
        self.binary.write_text(f"#!{sys.executable}\n" + body)
        self.binary.chmod(0o700)

    def measure(self, timeout=5):
        return sampler.measure(self.binary, self.model, "ministral", False, self.output, timeout)

    def assert_child_reaped(self):
        self.assertEqual(len(self.children), 1)
        self.assertIsNotNone(self.children[0].returncode)
        with self.assertRaises(ChildProcessError):
            os.waitpid(self.children[0].pid, os.WNOHANG)

    def test_success_writes_valid_records_and_preserves_prior_run(self):
        data = records()
        lines = "\n".join("REAL_MAC_JSON " + json.dumps(record) for record in data)
        self.script(f"print({lines!r})\n")
        self.assertEqual(self.measure(), 0)
        path = self.output / "run.json"
        original = path.read_bytes()
        result = json.loads(original)
        self.assertEqual(result["records"], data)
        self.assertEqual(result["returncode"], 0)
        self.assertGreater(result["elapsed_s"], 0)
        self.assertGreater(result["maxrss_bytes"], 0)
        self.assertTrue(all(s["rss_bytes"] == 123 * 1024 for s in result["samples"]))
        with self.assertRaises(FileExistsError):
            self.measure()
        self.assertEqual(path.read_bytes(), original)

    def test_empty_success_does_not_produce_measurement(self):
        self.script("print('PASS')\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.measure()
        self.assertFalse((self.output / "run.json").exists())
        self.assertIn("PASS", (self.output / "run.log").read_text())

    def test_failed_run_log_is_preserved(self):
        self.output.mkdir()
        log = self.output / "run.log"
        log.write_text("previous failure\n")
        with patch.object(sampler.subprocess, "Popen") as popen:
            with self.assertRaises(FileExistsError):
                self.measure()
            popen.assert_not_called()
        self.assertEqual(log.read_text(), "previous failure\n")

    def test_log_creation_is_exclusive_after_racing_existence_check(self):
        self.output.mkdir()
        log = self.output / "run.log"
        log.write_text("other sampler\n")
        with patch.object(Path, "exists", return_value=False):
            with self.assertRaises(FileExistsError):
                self.measure()
        self.assertEqual(self.children, [])
        self.assertEqual(log.read_text(), "other sampler\n")

    def test_failed_child_retains_diagnostics(self):
        self.script("print('fixture failure')\nraise SystemExit(7)\n")
        self.assertEqual(self.measure(), 7)
        result = json.loads((self.output / "run.json").read_text())
        self.assertEqual(result["returncode"], 7)
        self.assertIn("fixture failure", (self.output / "run.log").read_text())

    def test_timeout_kills_and_reaps_child(self):
        self.script("import time\ntime.sleep(60)\n")
        with self.assertRaises(TimeoutError):
            self.measure(timeout=0.05)
        self.assert_child_reaped()
        self.assertFalse((self.output / "run.json").exists())

    def test_stalled_telemetry_kills_and_reaps_child(self):
        self.script("import time\ntime.sleep(60)\n")
        with patch.object(sampler.subprocess, "check_output", side_effect=subprocess.TimeoutExpired("ps", 0.05)):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.measure(timeout=0.05)
        self.assert_child_reaped()
        self.assertFalse((self.output / "run.json").exists())


if __name__ == "__main__":
    unittest.main()
