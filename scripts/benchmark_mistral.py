#!/usr/bin/env python3
"""Compare two CLI builds with fresh prompt caches and identical greedy output."""
import argparse
import json
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True, help='Baseline CLI binary')
    parser.add_argument('--after', required=True, help='Candidate CLI binary')
    parser.add_argument('--model', required=True, help='Local GGUF checkpoint')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--metal', action='store_true', help='Enable Metal in both CLI builds')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--prompt', default='Erkläre in fünf Sätzen, wie ein Computerprogramm ausgeführt wird.')
    parser.add_argument('--timeout', type=int, default=300, help='Per-process timeout in seconds, including loading')
    args = parser.parse_args()
    if min(args.threads, args.runs, args.max_tokens, args.timeout) < 1:
        parser.error('threads, runs, max-tokens and timeout must be positive')
    results = {'before': [], 'after': []}
    process_wall_ms = {'before': [], 'after': []}
    reference = None
    # Alternate order to reduce warm-cache/order bias. Each process loads once;
    # its first run measures uncached prefill, its second cached-prompt decode.
    for run in range(args.runs):
        for name in (('before', 'after') if run % 2 == 0 else ('after', 'before')):
            command = [getattr(args, name), args.model, '--prompt', args.prompt,
                       '--max-tokens', str(args.max_tokens), '--temp', '0',
                       '--threads', str(args.threads), '--bench-json', '--bench-runs', '2']
            if args.metal:
                command.append('--metal')
            started = time.perf_counter()
            proc = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
            process_wall_ms[name].append((time.perf_counter() - started) * 1000)
            if proc.returncode:
                raise RuntimeError(f'{name} failed: {proc.stderr}')
            rows = json.loads(proc.stdout)
            if len(rows) != 2:
                raise RuntimeError(f'{name}: expected two benchmark results')
            for row in rows:
                signature = (row['prompt_tokens'], row['generated_tokens'], row['text'])
                if reference is None:
                    reference = signature
                elif signature != reference:
                    raise RuntimeError(f'{name}: generated text or token counts differ')
            results[name].append(rows)
    summary = {}
    for name, pairs in results.items():
        summary[name] = {
            'uncached_prefill_ms': statistics.median(p[0]['prefill_ms'] for p in pairs),
            'cached_decode_ms': statistics.median(p[1]['decode_ms'] for p in pairs),
            'cached_tokens_per_second': statistics.median(p[1]['generated_tokens_per_second'] for p in pairs),
            'process_wall_ms': statistics.median(process_wall_ms[name]),
        }
    print(json.dumps({'summary': summary, 'identical_output': True, 'runs': results,
                      'process_wall_ms': process_wall_ms}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
