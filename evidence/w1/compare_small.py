"""Summarize completed Small cells without relabelling missing cells."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/root/opset-research/w1-weight-layout-20261026')
B0 = Path('/root/opset-research/b0-weight-layout-20261023')
W1 = ROOT / 'w1-small-r1'


def read(path):
    return json.loads(path.read_text())


def main():
    output = {'baseline': str(B0), 'candidate': str(W1), 'cells': []}
    for bucket in range(1, 5):
        for concurrency in [1, 8, 32]:
            name = f'small-P{bucket}-C{concurrency}'
            a, b = B0/'cells'/name/'summary.json', W1/'cells'/name/'summary.json'
            if not b.exists():
                output['cells'].append({'name': name, 'status': 'pending'})
                continue
            baseline, candidate = read(a), read(b)
            row = {'name': name, 'status': candidate['status'],
                   'baseline_file': str(a), 'candidate_file': str(b),
                   'baseline_sha256': hashlib.sha256(a.read_bytes()).hexdigest(),
                   'candidate_sha256': hashlib.sha256(b.read_bytes()).hexdigest()}
            if baseline.get('status') == candidate.get('status') == 'completed':
                av = baseline['dual_card_output_tokens_per_second']
                bv = candidate['dual_card_output_tokens_per_second']
                row.update(b0_tokens_s=av, w1_tokens_s=bv, ratio=bv/av,
                    change_pct=(bv/av-1)*100,
                    requests=candidate['requests'], failures=candidate['failures'],
                    output_tokens=candidate['output_tokens'],
                    b0_ttft=baseline['ttft_seconds'], w1_ttft=candidate['ttft_seconds'],
                    b0_tpot=baseline['tpot_seconds'], w1_tpot=candidate['tpot_seconds'])
            output['cells'].append(row)
    complete = [r for r in output['cells'] if 'ratio' in r]
    output['completed_performance_cells'] = len(complete)
    output['geomean_change_pct'] = ((math.prod(r['ratio'] for r in complete)**(1/len(complete))-1)*100
                                  if complete else None)
    for arm, root in [('B0', B0), ('W1', W1)]:
        p = root/'cells/small-quality/summary.json'
        if p.exists():
            output[arm+'_quality'] = read(p).get('quality')
    qa, qb = B0/'cells/small-quality/requests.jsonl', W1/'cells/small-quality/requests.jsonl'
    if qa.exists() and qb.exists():
        aa = {r['id']: r for r in map(json.loads, qa.read_text().splitlines())}
        bb = {r['id']: r for r in map(json.loads, qb.read_text().splitlines())}
        output['quality_changes'] = [{'id': i, 'b0': aa[i].get('text'), 'w1': bb[i].get('text')}
            for i in sorted(aa.keys() & bb.keys()) if aa[i].get('text') != bb[i].get('text')]
    (ROOT/'small-comparison.json').write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
