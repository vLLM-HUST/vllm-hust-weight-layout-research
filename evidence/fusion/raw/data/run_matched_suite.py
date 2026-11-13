"""Three independent service lifecycles per arm; fail closed quality gate."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
record=dict(status='running',started=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()),runs=[],contract=dict(rounds=3,order=['baseline/fusion','fusion/baseline','baseline/fusion'],quality='same first 20 pinned MMLU-Pro rows; candidate accuracy >= matched baseline, no degenerate output',timing='3 warmups per workload, 5 C1 samples and 3 C2 pairs; startup and profiler excluded',numerical='42 changing-input FP16/BF16 Graph comparisons must be finite and close; observed exact equality',scope='development sample; not full five-dataset certification'))

def write(): root.joinpath('matched-suite.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')

write()
try:
    validation=json.loads(root.joinpath('results/shared-rounded-validation.json').read_text())
    if not validation['all_pass']: raise RuntimeError('operator correctness gate failed')
    for iteration in range(1,4):
        pair={}
        for variant in (['baseline','fusion'] if iteration!=2 else ['fusion','baseline']):
            command=[sys.executable,str(root/'lifecycle_benchmark.py'),'--variant',variant,'--round',str(iteration)]
            if iteration==3: command+=['--profile']
            log=root/'results'/f'matched-{iteration}-{variant}.log'
            with log.open('w') as output:
                returncode=subprocess.call(command,stdout=output,stderr=subprocess.STDOUT,timeout=1800)
            state=json.loads(root.joinpath('suite-current.json').read_text())
            record['runs'].append(state); pair[variant]=state; write()
            if returncode or state['status']!='completed': raise RuntimeError('lifecycle failed: '+state['run_id'])
        if pair['fusion']['quality_accuracy']<pair['baseline']['quality_accuracy']:
            raise RuntimeError('sampled quality regression; stop performance claim')
    record['status']='finished'
except Exception as error:
    record.update(status='failed',error=f'{type(error).__name__}: {error}')
finally: write()
print(json.dumps(record,ensure_ascii=False),flush=True)
