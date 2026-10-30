#!/usr/bin/env python3
"""Run CHANNEL correctness/lifecycle contracts sequentially on the local SM120 GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    args.install = args.install.resolve(strict=True)
    args.build = args.build.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('XSCHED_') and k not in ('LD_PRELOAD', 'LD_LIBRARY_PATH')}
    env.update(XSCHED_SCHEDULER='APP', XSCHED_CUDA_LV3_IMPL='CHANNEL',
               LD_LIBRARY_PATH=str(args.install / 'lib'))
    tracker = args.install / 'lib/libcuda_channel_tracker.so'
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.used,temperature.gpu',
                          '--format=csv'], text=True, capture_output=True, check=True).stdout
    files = [tracker, args.install / 'bin/cuda/sm120_lv3', args.install / 'bin/cuda/sm120_channel_contract']
    metadata = dict(gpu=gpu, install=str(args.install), build=str(args.build), repeats=args.repeats,
                    sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    cases = [('tracker-unit', [str(args.build / 'platforms/cuda/cuda_channel_tracker_test')], False)]
    for repeat in range(args.repeats):
        cases.extend([
            (f'lv3-r{repeat}', [str(args.install / 'bin/cuda/sm120_lv3')], True),
            (f'contract-r{repeat}', [str(args.install / 'bin/cuda/sm120_channel_contract')], True),
        ])
    cases.extend([
        ('duplicate-owner', [str(args.install / 'bin/cuda/sm120_lv3'), 'duplicate'], True),
        ('missing-tracker', [str(args.install / 'bin/cuda/sm120_channel_contract'), 'missing-tracker'], False),
    ])
    results = []
    for name, command, preload in cases:
        case_env = dict(env)
        if preload:
            case_env['LD_PRELOAD'] = str(tracker)
        result = subprocess.run(['timeout', '--kill-after=3s', '60s', *command], env=case_env,
                                text=True, capture_output=True)
        (args.output / f'{name}.stdout.log').write_text(result.stdout)
        (args.output / f'{name}.stderr.log').write_text(result.stderr)
        passed = result.returncode == 0 and 'PASS' in result.stdout
        results.append(dict(case=name, passed=passed, returncode=result.returncode, command=command))
        (args.output / 'summary.json').write_text(json.dumps(results, indent=2))
        if not passed:
            raise RuntimeError(f'{name}: exit={result.returncode}; inspect saved logs')
        print(f'PASS {name}', flush=True)


if __name__ == '__main__':
    main()
