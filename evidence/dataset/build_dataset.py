import collections
import gzip
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

SOURCE = Path('/root/opset-research/datasets/mmlu-pro')
MODEL = Path('/data/shared_upload/models/vspec/Qwen3.5-35B-A3B')
OUT = SOURCE.parent / 'mmlu-weight-layout-v1'
SEED = 20261023


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_rows(path, rows):
    count = 0
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=6) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
            count += 1
    return count


def question(r):
    return r['question'] + '\n' + '\n'.join(
        f'{chr(65+i)}. {v}' for i, v in enumerate(r['options']))


def main():
    OUT.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    data = {s: pq.read_table(next(SOURCE.glob(s+'-*.parquet'))).to_pylist()
            for s in ['validation', 'test']}
    metadata = {'version': 'mmlu-weight-layout-v1', 'seed': SEED,
                'model_path': str(MODEL), 'performance_source': 'validation only',
                'performance_encoding': 'canonical prompt_token_ids; use token-ID input',
                'quality_protocol': {'temperature': 0, 'seed': SEED,
                                     'max_tokens': 16, 'ignore_eos': False,
                                     'enable_thinking': False,
                                     'answer_parser': 'strip whitespace; accept exactly one valid uppercase option letter; otherwise invalid'},
                'performance_protocol': {'temperature': 0, 'seed': SEED,
                                         'ignore_eos': True, 'max_tokens': 'per record output_tokens',
                                         'prefix_caching': False, 'mtp': False},
                'sources': {}, 'tokenizer_files': {}, 'sets': {}}
    for p in SOURCE.glob('*.parquet'):
        metadata['sources'][p.name] = {'sha256': digest(p), 'bytes': p.stat().st_size}
    for name in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json']:
        p = MODEL / name
        if p.exists():
            metadata['tokenizer_files'][name] = digest(p)
    pool = {r['question_id']: tokenizer.encode(question(r), add_special_tokens=False)
            for r in data['validation']}
    separator = tokenizer.encode('\n\n', add_special_tokens=False)
    marker = 'MMLU_PAYLOAD_MARKER_20261023'
    for split, source, per_bucket in [('small', 'validation', 32), ('large', 'test', 256)]:
        folder = OUT / split
        folder.mkdir()
        rows = sorted(data[source], key=lambda r: r['question_id'])
        def quality():
            for r in rows:
                messages = [{'role': 'user', 'content':
                    'Answer the following multiple-choice question. Output only the uppercase letter of the correct option, with no explanation.\n\n' + question(r)}]
                ids = tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_dict=False)
                yield {'id': f"{source}-{r['question_id']}", 'source_split': source,
                       'question_id': r['question_id'], 'category': r['category'],
                       'messages': messages, 'prompt_token_ids': ids,
                       'input_tokens': len(ids), 'output_tokens': 16,
                       'num_options': len(r['options'])}
        nq = write_rows(folder / 'quality.requests.jsonl.gz', quality())
        write_rows(folder / 'quality.answers.jsonl.gz', (
            {'id': f"{source}-{r['question_id']}", 'answer': r['answer'],
             'answer_index': r['answer_index'], 'category': r['category']} for r in rows))
        buckets = [(512, 128), (1024, 1024), (8192, 256), (32768, 256)]
        if split == 'large':
            buckets.append((131072, 256))
        def performance():
            for bucket, (length, output) in enumerate(buckets, 1):
                for j in range(per_bucket):
                    rid = f'{split}-P{bucket}-{j:04d}'
                    rng = random.Random(f'{SEED}:{rid}')
                    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content':
                        f'Benchmark case {rid}. Read the following multiple-choice material and discuss its subjects and reasoning in detail.\n\n{marker}'}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    before, after = rendered.split(marker)
                    prefix = tokenizer.encode(before, add_special_tokens=False)
                    suffix = tokenizer.encode(after, add_special_tokens=False)
                    budget = length - len(prefix) - len(suffix)
                    body, provenance = [], []
                    while len(body) < budget:
                        order = sorted(pool)
                        rng.shuffle(order)
                        for qid in order:
                            start = len(body)
                            piece = pool[qid] + separator
                            take = min(budget-start, len(piece))
                            body.extend(piece[:take])
                            provenance.append({'question_id': qid, 'tokens_used': take})
                            if len(body) == budget:
                                break
                    ids = prefix + body + suffix
                    assert len(ids) == length
                    yield {'id': rid, 'bucket': f'P{bucket}', 'source_split': 'validation',
                           'synthetic': True, 'quality_scored': False,
                           'input_tokens': length, 'output_tokens': output,
                           'prompt_token_ids': ids,
                           'token_ids_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
                           'source_segments': provenance}
        np = write_rows(folder / 'performance.requests.jsonl.gz', performance())
        metadata['sets'][split] = {'quality_count': nq, 'performance_count': np,
            'categories': dict(collections.Counter(r['category'] for r in rows)),
            'buckets': [{'name': f'P{i+1}', 'input_tokens': a, 'output_tokens': b,
                         'requests': per_bucket, 'min_context_tokens': a+b}
                        for i, (a,b) in enumerate(buckets)]}
        print(split, nq, np, flush=True)
    verification = {}
    all_ids = set()
    for p in OUT.glob('*/*.jsonl.gz'):
        count = 0
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if 'prompt_token_ids' in r:
                    assert len(r['prompt_token_ids']) == r['input_tokens']
                    assert r['id'] not in all_ids
                    all_ids.add(r['id'])
                    assert 'answer' not in r and 'cot_content' not in r
                if r.get('synthetic'):
                    assert r['source_split'] == 'validation'
                    assert all(s['question_id'] in pool for s in r['source_segments'])
                    h = hashlib.sha256(json.dumps(r['prompt_token_ids'],separators=(',', ':')).encode()).hexdigest()
                    assert h == r['token_ids_sha256']
                count += 1
        verification[str(p.relative_to(OUT))] = {'records': count, 'sha256': digest(p), 'bytes': p.stat().st_size}
    metadata['files'] = verification
    metadata['verification'] = {'passed': True, 'checks': ['gzip JSON readability', 'unique request IDs',
        'exact input token counts', 'performance token hashes', 'validation-only performance provenance',
        'no answer or cot_content fields in requests']}
    (OUT / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata['sets']), flush=True)
    print('VERIFIED', OUT, flush=True)


if __name__ == '__main__':
    main()
