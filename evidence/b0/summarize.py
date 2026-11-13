"""Export measured B0 data, metrics deltas, curves and a readable report."""
import collections
import csv
import datetime
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parent


def metric(path, prefix):
    if not path.exists():return None
    values=[]
    for l in path.read_text().splitlines():
        if l.startswith(prefix+'{') or l.startswith(prefix+' '):
            try:values.append(float(l.rsplit(' ',1)[1]))
            except ValueError:pass
    return sum(values) if values else None


def main():
    out=ROOT/'report';out.mkdir(exist_ok=True)
    metric_samples=[]
    if (ROOT/'metrics-samples.jsonl').exists():
        for line in (ROOT/'metrics-samples.jsonl').read_text().splitlines():
            try:metric_samples.append(json.loads(line))
            except json.JSONDecodeError:pass
    memory_samples=[]
    if (ROOT/'npu-samples.jsonl').exists():
        for line in (ROOT/'npu-samples.jsonl').read_text().splitlines():
            try:memory_samples.append(json.loads(line))
            except json.JSONDecodeError:pass
    rows=[];details={}
    for p in sorted((ROOT/'cells').glob('*/summary.json')):
        d=json.loads(p.read_text());name=p.parent.name
        details[name]=d
        spec=json.loads((p.parent/'spec.json').read_text()) if (p.parent/'spec.json').exists() else {}
        match=re.search(r'(?:^|-)(P\d)-C(\d+)',name)
        repeat=re.search(r'(?:large|medium)-r(\d+)',name)
        before=metric(p.parent/'metrics.before.txt','vllm:num_preemptions_total')
        after=metric(p.parent/'metrics.after.txt','vllm:num_preemptions_total')
        # Older Prometheus builds may retain the unsuffixed counter name.
        if before is None:before=metric(p.parent/'metrics.before.txt','vllm:num_preemptions')
        if after is None:after=metric(p.parent/'metrics.after.txt','vllm:num_preemptions')
        row={'cell':name,'bucket':match[1] if match else '',
             'concurrency':int(match[2]) if match else spec.get('concurrency',''),
             'repeat':int(repeat[1]) if repeat else '', 'status':d['status'],
             'requests':d.get('requests',spec.get('requests','')),
             'successes':d.get('successes',''),'failures':d.get('failures',''),
             'seconds':d.get('seconds',''),'output_tokens_per_second':d.get('dual_card_output_tokens_per_second',''),
             'input_tokens_per_second':d.get('input_tokens_per_second',''),
             'requests_per_second':d.get('requests_per_second',''),
             'preemptions':after-before if before is not None and after is not None else ''}
        # Explicitly separate unmeasured cells from actual timed request failures.
        row['planned_requests']=spec.get('requests','')
        row['warmup_failures']=len(d.get('warmup_errors',[]))
        if d['status']=='warmup_failed':
            row['requests']=0;row['successes']=0;row['failures']=0
        observed_running=[];observed_waiting=[];observed_kv=[];hbm={6:[],7:[]}
        raw=p.parent/'requests.jsonl'
        begin=end=None
        if raw.exists():
            raw_rows=[]
            for line in raw.read_text().splitlines():
                try:raw_rows.append(json.loads(line))
                except json.JSONDecodeError:pass
            if raw_rows:
                begin=min(datetime.datetime.fromisoformat(x['started_utc']).timestamp() for x in raw_rows)
                end=max(datetime.datetime.fromisoformat(x['started_utc']).timestamp()+x['e2e_seconds'] for x in raw_rows)
        if begin is not None:
            for sample in metric_samples:
                tick=datetime.datetime.fromisoformat(sample['utc']).timestamp()
                if not begin<=tick<=end:continue
                for key,val in sample['metrics'].items():
                    if key.startswith('vllm:num_requests_running{'):observed_running.append(val)
                    elif key.startswith('vllm:num_requests_waiting{'):observed_waiting.append(val)
                    elif key.startswith('vllm:kv_cache_usage_perc{'):observed_kv.append(val)
            for sample in memory_samples:
                if 'utc' not in sample:continue
                tick=datetime.datetime.fromisoformat(sample['utc']).timestamp()
                if not begin<=tick<=end:continue
                lines=sample.get('output','').splitlines()
                for i,line in enumerate(lines):
                    m=re.search(r'\|\s*(6|7)\s+910B2',line)
                    if m and i+1<len(lines):
                        u=re.search(r'(\d+)\s*/\s*65536',lines[i+1])
                        if u:hbm[int(m[1])].append(int(u[1]))
        row['observed_running_max_5s']=max(observed_running,default='')
        row['observed_waiting_max_5s']=max(observed_waiting,default='')
        row['observed_kv_usage_fraction_max_5s']=max(observed_kv,default='')
        for card in [6,7]:row[f'npu{card}_hbm_mib_observed_peak_5s']=max(hbm[card],default='')
        for k in ['ttft_seconds','tpot_seconds','e2e_seconds']:
            for q in ['mean','p50','p95','p99']:row[k+'_'+q]=d.get(k,{}).get(q,'')
        rows.append(row)
    if rows:
        with (out/'cells.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    aggregates=[]
    for cohort,bucket in [(s,b) for s in ['medium','large'] for b in ['P1','P2','P3','P4','P5']]:
        for c in [1,4,8,16,32,64]:
            a=[r for r in rows if r['cell'].startswith(cohort+'-r') and r['bucket']==bucket and r['concurrency']==c and r['repeat']!='']
            good=[r for r in a if r['status']=='completed']
            x=[r['output_tokens_per_second'] for r in good]
            aggregates.append({'cohort':cohort,'bucket':bucket,'concurrency':c,'successful_repeats':len(good),
                'tested_repeats':len(a),'output_tokens_per_second_mean':statistics.mean(x) if x else None,
                'output_tokens_per_second_stdev':statistics.stdev(x) if len(x)>1 else None,
                'max_ttft_p95':max((r['ttft_seconds_p95'] for r in good),default=None),
                'max_tpot_p95':max((r['tpot_seconds_p95'] for r in good),default=None)})
    (out/'aggregates.json').write_text(json.dumps(aggregates,indent=2))
    samples=[]
    starts=[json.loads(p.read_text()).get('started_utc','') for p in (ROOT/'servers').glob('*/launch.json')]
    formal_start=min(starts) if starts else ''
    samplefile=ROOT/'npu-samples.jsonl'
    if samplefile.exists():
        for line in samplefile.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r.get('utc','')<formal_start:continue
            lines=r.get('output','').splitlines()
            for i,l in enumerate(lines):
                m=re.search(r'\|\s*(\d+)\s+910B2',l)
                if not m or i+1>=len(lines):continue
                usage=re.search(r'(\d+)\s*/\s*65536',lines[i+1])
                if usage:samples.append({'utc':r.get('utc'),'npu':int(m[1]),'hbm_mib':int(usage[1]),
                    'cell':r.get('stage',{}).get('cell'), 'stage':r.get('stage',{}).get('stage')})
    if samples:
        with (out/'hbm-samples.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(samples[0]));w.writeheader();w.writerows(samples)
    evidence={}
    for p in (ROOT/'servers').glob('*/server.log'):
        lines=p.read_text(errors='replace').splitlines()
        evidence[p.parent.name]={
            'memory':[l for l in lines if re.search(r'Loading model weights took|Available KV cache|GiB for|graph.*GiB|Graph capturing finished',l,re.I)],
            'graph':[l for l in lines if re.search(r'CUDAGraph|cudagraph|Capturing CUDA graphs',l)][-100:],
            'binding':[l for l in lines if re.search(r'cpu_bind|Bind cpus|CPU binding|cpu_global_slice',l)],
            'compilation_or_fallback':[l for l in lines if re.search(r'recompil|fallback|falling back|Compiling model again',l,re.I)]}
    (out/'server-evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    if any(r['status']=='completed' and r['bucket'] for r in rows):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for ax,bucket in zip(axes.flat,['P1','P2','P3','P4','P5']):
            for cohort in ['medium','large']:
                a=[x for x in aggregates if x['cohort']==cohort and x['bucket']==bucket and x['output_tokens_per_second_mean'] is not None]
                if a:
                    ax.errorbar([x['concurrency'] for x in a],[x['output_tokens_per_second_mean'] for x in a],
                        yerr=[x['output_tokens_per_second_stdev'] or 0 for x in a],marker='o',capsize=3,label=cohort.title()+' mean +/- SD')
            small=[r for r in rows if r['cell'].startswith('small-') and r['bucket']==bucket and r['status']=='completed']
            small.sort(key=lambda r:r['concurrency'])
            if small:ax.plot([x['concurrency'] for x in small],[x['output_tokens_per_second'] for x in small],
                             '--o',label='Small screening')
            ax.set_title(bucket);ax.set_xlabel('Client concurrency');ax.set_ylabel('Dual-NPU output tokens/s')
            ax.grid(alpha=.3)
            if ax.lines:ax.legend(fontsize=8)
        axes.flat[-1].axis('off');fig.tight_layout()
        fig.savefig(out/'throughput-curves.png',dpi=160);plt.close(fig)
    statuses=dict(collections.Counter(d['status'] for d in details.values()))
    text=['# B0 实验结果','',f'已记录单元状态：`{statuses}`。所有数值来自原始结果；未完成单元不填推测值。','',
          '默认 Small + Medium；Large 仅保留历史测量。不同规模独立汇总，不混合平均。', '',
          '| 数据集/档位 | 并发 | 成功重复/已测重复 | 双卡输出 tokens/s 均值 | 标准差 |',
          '| --- | ---: | ---: | ---: | ---: |']
    for a in aggregates:
        value=a['output_tokens_per_second_mean'];sd=a['output_tokens_per_second_stdev']
        text.append(f"| {a['cohort']}/{a['bucket']} | {a['concurrency']} | {a['successful_repeats']}/{a['tested_repeats']} | {value:.2f} | {sd:.2f} |" if value is not None and sd is not None else
                    f"| {a['cohort']}/{a['bucket']} | {a['concurrency']} | {a['successful_repeats']}/{a['tested_repeats']} | {value if value is not None else ('未测试' if not a['tested_repeats'] else '未取得成功结果')} | — |")
    for name in ['small-quality','medium-quality','large-quality']:
        if name in details:text+=['',f'## {name}','', '```json',json.dumps(details[name].get('quality',details[name]),ensure_ascii=False,indent=2),'```']
    text+=['','## 指标与限制','',
        'TTFT/TPOT 是客户端 SSE 观测值。输入吞吐以整个请求批次墙钟为分母，不是隔离的纯 Prefill 算子吞吐。',
        'HBM 为5秒采样的观测峰值，不等价于瞬时 allocator 峰值。独立工作区未能从日志拆分时不编造数值。',
        '细分结果见 cells.csv；图覆盖与显存日志见 server-evidence.json；Profiler 与15分钟稳定性原始数据分别在 profiles/ 和 stability/。']
    (out/'RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(statuses))


if __name__=='__main__':main()
