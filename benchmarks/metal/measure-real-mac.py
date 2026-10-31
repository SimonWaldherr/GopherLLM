import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser(description='Measure one opt-in real Mac decoder run with process RSS and global VM sampling.')
parser.add_argument('--binary', required=True, type=Path)
parser.add_argument('--model', required=True, type=Path)
parser.add_argument('--kind', required=True, choices=['ministral', 'voxtral'])
parser.add_argument('--copy', action='store_true', help='Use the prior copying Voxtral loader as control')
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--timeout', type=float, default=300)
args = parser.parse_args()
base = args.output.resolve()
base.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]

def validate_records(records, kind):
    decodes = [r for r in records if r.get("phase") == "decode"]
    expected = 3 if kind == "voxtral" else 4
    if not any(r.get("phase") == "loaded" for r in records) or len(decodes) != expected:
        raise ValueError("incomplete decoder measurements; check the test binary and log")
    if any(r.get("tokens") != 64 or r.get("seconds", 0) <= 0 for r in decodes):
        raise ValueError("unexpected decode length or duration")


def vm():
    s = subprocess.check_output(['/usr/bin/vm_stat'], text=True)
    d = {k: int(v) for k, v in re.findall('^([^:\\n]+):\\s+(\\d+)\\.', s, re.M)}
    return {'page_size': int(re.search('page size of (\\d+)', s)[1]), 'counters': d, 'swapusage': subprocess.check_output(['/usr/sbin/sysctl', 'vm.swapusage'], text=True).strip()}
name, kind, path, copy = 'run', args.kind, str(args.model.resolve()), args.copy
before = vm()
started = time.time()
deadline = time.monotonic() + args.timeout
samples = []
with (base / (name + '.log')).open('w') as f:
    p = subprocess.Popen([str(args.binary.resolve()), '-test.run', '^TestRealMacDecode$', '-test.v', '-test.count=1'], cwd=root / 'internal/engine', env=dict(os.environ, GOMAXPROCS='8', GOPHERLLM_REAL_MAC_KIND=kind, GOPHERLLM_REAL_MAC_MODEL=path, GOPHERLLM_REAL_MAC_COPY='1' if copy else '0'), stdout=f, stderr=subprocess.STDOUT)
    try:
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError('model measurement timed out')
            pid, status, ru = os.wait4(p.pid, os.WNOHANG)
            if pid:
                p.returncode = os.waitstatus_to_exitcode(status)
                break
            try:
                rss = int(subprocess.check_output(['/bin/ps', '-o', 'rss=', '-p', str(p.pid)], text=True).strip()) * 1024
            except (subprocess.CalledProcessError, ValueError):
                rss = None
            samples.append({'unix_time': time.time(), 'rss_bytes': rss, 'vm': vm()})
            time.sleep(0.5)
    except BaseException:
        if p.returncode is None:
            p.kill()
            p.wait()
        raise
after = vm()
records = []
for line in (base / (name + '.log')).read_text().splitlines():
    if line.startswith('REAL_MAC_JSON '):
        records.append(json.loads(line.split(' ', 1)[1]))
if p.returncode == 0:
    validate_records(records, kind)
out = {'kind': kind, 'loader_copy': copy, 'model': Path(path).name, 'mode': name, 'started': started, 'elapsed_s': time.time() - started, 'returncode': p.returncode, 'before': before, 'after': after, 'records': records, 'samples': samples, 'maxrss_bytes': ru.ru_maxrss, 'major_faults': ru.ru_majflt, 'minor_faults': ru.ru_minflt}
(base / (name + '.json')).write_text(json.dumps(out, indent=2))
print(name, 'exit', p.returncode, 'seconds', out['elapsed_s'], 'maxrssGiB', ru.ru_maxrss / 2 ** 30, flush=True)
if p.returncode:
    print((base / (name + '.log')).read_text()[-2500:], flush=True)
    raise SystemExit(p.returncode)
