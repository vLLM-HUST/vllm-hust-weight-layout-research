"""Mini paired confirmation, only after a positive W1 Small screen."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

BASE = Path('/root/opset-research/w1-weight-layout-20261026')


async def main(arm, repeat, include_p4):
    import fcntl
    lock = (BASE / 'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = subprocess.check_output(['npu-smi', 'info'], text=True)
    if state.count('No running processes found in NPU') != 2:
        raise RuntimeError('NPUs occupied')
    root = BASE / f'{arm.lower()}-mini-r{repeat}'
    root.mkdir(exist_ok=False)
    os.environ.update(W1_RUN_ROOT=str(root), W1_ARM=arm, B0_DEVICES='0,1')
    spec = importlib.util.spec_from_file_location('client', BASE / 'client_snapshot.py')
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    import signal
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    buckets = ['P2', 'P3'] + (['P4'] if include_p4 else [])
    try:
        for bucket in buckets:
            rows = c.records('small', 'performance', bucket)[:16]
            c.save(root / f'{bucket}.selection.json', {
                'parent': 'small/performance.requests.jsonl.gz',
                'parent_sha256': c.sha(c.DATA / 'small/performance.requests.jsonl.gz'),
                'ids': [r['id'] for r in rows],
                'selection': 'first16 of unchanged B0 seed-shuffled Small bucket'})
            server = c.Server(f'{arm}-mini-r{repeat}-{bucket}', c.CONTEXT[bucket], 32, 4096)
            try:
                await server.start()
                for concurrency in ([16] if bucket == 'P4' else [1, 16]):
                    result = await c.batch(f'mini-{bucket}-C{concurrency}', rows, concurrency, False)
                    if result.get('status') != 'completed':
                        raise RuntimeError('Mini cell failed')
            finally:
                await server.stop()
        c.summarize()
    finally:
        (root / 'npu.final.txt').write_text(subprocess.check_output(['npu-smi', 'info'], text=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['B0', 'W1'], required=True)
    p.add_argument('--repeat', type=int, choices=[1, 2], required=True)
    p.add_argument('--include-p4', action='store_true')
    a = p.parse_args()
    asyncio.run(main(a.arm, a.repeat, a.include_p4))
