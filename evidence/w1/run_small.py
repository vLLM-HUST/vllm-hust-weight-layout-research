"""Prepared Small-only paired campaign. No execution without --execute."""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess

BASE = Path('/root/opset-research/w1-weight-layout-20261026')


async def execute(arm, repeat):
    import fcntl
    BASE.mkdir(exist_ok=True)
    lock = (BASE / 'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Refuse a used API endpoint; never attach to or stop an unrelated server.
    probe = socket.socket()
    try:
        probe.bind(('127.0.0.1', 18124))
    finally:
        probe.close()
    state = subprocess.check_output(['npu-smi', 'info'], text=True)
    if state.count('No running processes found in NPU') != 2:
        raise RuntimeError('Both NPUs must be idle before starting; no processes stopped')
    out = BASE / f'{arm.lower()}-small-r{repeat}'
    out.mkdir(exist_ok=False)  # preserve prior attempts; never silently overwrite
    (out / 'npu.before.txt').write_text(state)
    os.environ['W1_RUN_ROOT'] = str(out)
    os.environ['W1_ARM'] = arm
    os.environ['B0_DEVICES'] = '0,1'
    spec = importlib.util.spec_from_file_location('frozen_client', BASE / 'client_snapshot.py')
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    import signal
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    try:
        client.freeze()
        groups = [([1, 2], 4096), ([3], 16384), ([4], 65536)]
        if repeat == 2:
            groups.reverse()
        for buckets, context in groups:
            tasks = []
            if context == 4096:
                tasks.append(('small-quality', 'small', 'quality', None, 8))
            for bucket in buckets:
                conc = [32, 8, 1] if repeat == 2 else [1, 8, 32]
                tasks.extend((f'small-P{bucket}-C{c}', 'small', 'performance', f'P{bucket}', c) for c in conc)
            await client.group(f'{arm}-r{repeat}-ctx{context}', context, 32, 4096, tasks)
            results = client.summarize()
            if any(r.get('status') != 'completed' for r in results.values()):
                raise RuntimeError('Failed cell retained; stop this arm and diagnose on Small')
        client.save(out / 'execution.complete.json', {'arm': arm, 'repeat': repeat,
            'medium_allowed': False, 'note': 'Requires reviewed paired performance and quality gate'})
    finally:
        (out / 'npu.final.txt').write_text(subprocess.check_output(['npu-smi', 'info'], text=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['B0', 'W1'], required=True)
    p.add_argument('--repeat', type=int, choices=[1, 2, 3], required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if a.execute:
        asyncio.run(execute(a.arm, a.repeat))
    else:
        print(json.dumps({'arm': a.arm, 'repeat': a.repeat, 'dataset': 'small',
            'performance_cells': 12, 'quality_requests': 70, 'tp': 2,
            'ep': a.arm == 'W1', 'pp': 1, 'dp': 1, 'max_num_seqs': 32,
            'max_num_batched_tokens': 4096, 'execute': False,
            'medium_queue': False, 'large_queue': False}, indent=2))
