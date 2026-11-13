"""Audit and render final results from preserved artifacts; no remote workloads."""
import collections,datetime,json,re,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'report';OUT.mkdir(exist_ok=True)
s=json.loads((ROOT/'final-audit/final-evidence.json').read_text(encoding='utf-8'))
cells=s['cells'];CONC=[1,4,8,16,32,64]
def mean(rs,key):return statistics.mean(r[key] for r in rs)
def desc(rs,key,q):return statistics.mean(r[key][q] for r in rs)
def preempt(name):
    v=s['preemptions'][name]
    return v['after'].get('vllm:num_preemptions_total',0)-v['before'].get('vllm:num_preemptions_total',0)
aggregate=[]
for b in range(1,6):
    for c in CONC:
        rs=[cells[f'medium-r{r}-P{b}-C{c}'] for r in [1,2,3]]
        good=[r for r in rs if r['status']=='completed']
        row={'bucket':f'P{b}','concurrency':c,'success_repeats':len(good),
            'statuses':[r['status'] for r in rs]}
        if good:
            row.update(output_tps=mean(good,'dual_card_output_tokens_per_second'),
                output_tps_sd=statistics.stdev(r['dual_card_output_tokens_per_second'] for r in good),
                input_tps=mean(good,'input_tokens_per_second'),rps=mean(good,'requests_per_second'),
                ttft_mean=desc(good,'ttft_seconds','mean'),tpot_mean=desc(good,'tpot_seconds','mean'),
                e2e_mean=desc(good,'e2e_seconds','mean'))
            for metric in ['ttft_seconds','tpot_seconds','e2e_seconds']:
                row[metric]={q:desc(good,metric,q) for q in ['mean','p50','p95','p99']}
                row[metric]['worst_repeat_p99']=max(r[metric]['p99'] for r in good)
        aggregate.append(row)
(OUT/'medium-aggregates.json').write_text(json.dumps(aggregate,indent=2))
assert sum(r['success_repeats'] for r in aggregate)==87
assert sum(cells[f'medium-r{r}-P{b}-C{c}']['failures'] for r in [1,2,3] for b in range(1,6) for c in CONC if cells[f'medium-r{r}-P{b}-C{c}']['status']=='completed')==0
# These samples include warmup; keep that scope in the report.
hbm=collections.defaultdict(lambda:collections.defaultdict(int));running=collections.defaultdict(int)
for line in (ROOT/'final-audit/npu-samples.jsonl').read_text().splitlines():
    row=json.loads(line);cell=row.get('stage',{}).get('cell','')
    if not cell.startswith('medium-r'):continue
    ls=row.get('output','').splitlines()
    for i,l in enumerate(ls):
        m=re.search(r'\|\s*(\d+)\s+910B2',l)
        if m and i+1<len(ls):
            u=re.search(r'(\d+)\s*/\s*65536',ls[i+1])
            if u:hbm[cell][m[1]]=max(hbm[cell][m[1]],int(u[1]))
for line in (ROOT/'final-audit/metrics-samples.jsonl').read_text().splitlines():
    row=json.loads(line);cell=row.get('stage',{}).get('cell','')
    for k,v in row.get('metrics',{}).items():
        if k.startswith('vllm:num_requests_running{'):running[cell]=max(running[cell],v)
(OUT/'memory-observed.json').write_text(json.dumps({'hbm_peak_mib_5s_including_warmup':hbm,'max_running_5s_including_warmup':running},indent=2))
text=['# B0 双卡 TP2 基线实验结果','',
 '测试队列已结束，含已记录的失败点。全部完成时间：北京时间 **2026-10-24 00:47:06**。',
 'Medium 三轮90个单元：87成功，3个P5 C64预热读取超时；87个成功单元共5568条正式请求，正式请求失败0。失败单元没有正式计时结果，不能填0吞吐或算作192条正式失败。', '',
 '## 配置与可比性','',
 'Qwen3.5-35B-A3B，双Ascend 910B2（原实验物理卡6/7），BF16，TP2/DP1/PP1，EP关闭；固定源码、权重、tokenizer及数据hash，MTP/APC/异步调度关闭。',
 'Medium-v2：P1–P5每档64条，质量2406题。并发1/4/8/16/32/64，max-num-seqs64，max-num-batched-tokens8192，显存利用率0.85。输入/输出分别为512/128、1024/1024、8192/256、32768/256、131072/256。上下文上限4096/4096/16384/65536/262144。',
 '每个上下文三次独立启动，第二轮逆序。预热/编译排除计时；Profiler单独采集。Small沿用4096预算筛选结果；不能把Small与Medium的差异全部归因于样本规模。Large历史批吞吐独立保留，不参与Medium均值。',
 '容器实际限制8 CPU核配额、64GiB主存。API/Engine/TP0/TP1分别绑定24–31/32–39/96–119/48–71，内存交错NUMA2/4；一个API内核iou-sqp线程无法改亲和性并已记录。', '',
 '## 三轮平均双卡输出吞吐','',
 '单位tokens/s；括号内为三轮样本标准差。每档64条短批包含排队和收尾效应。', '',
 '| 档位 | C1 | C4 | C8 | C16 | C32 | C64 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for b in range(1,6):
    rows=[r for r in aggregate if r['bucket']==f'P{b}']
    text.append('| P'+str(b)+' | '+' | '.join(f"{r['output_tps']:.2f} ({r['output_tps_sd']:.2f})" if r['success_repeats'] else '3轮预热超时' for r in rows)+' |')
text+=['','![吞吐随并发变化](throughput-final.png)','',
 'P2 C64/C1平均批吞吐比约25.2倍。P4 C32→64增益约1.9%；P5 C16→32增益约0.2%，已接近该配置的吞吐平台。P1 C32标准差较大，应视为波动点，不能从单轮作确定性优化结论。', '',
 '## 已复用的Small筛选结果','',
 'Small预算4096、max-num-seqs32，以下是原筛选单轮结果，不能与Medium三轮均值作同配置收益比较。单位：双卡输出tokens/s。','',
 '| 档位 | C1 | C8 | C32 |','| --- | ---: | ---: | ---: |']
for b in range(1,5):
    text.append('| P'+str(b)+' | '+' | '.join(f"{cells[f'small-P{b}-C{c}']['dual_card_output_tokens_per_second']:.2f}" for c in [1,8,32])+' |')
text+=['','## 低并发 Decode','']
r=next(x for x in aggregate if x['bucket']=='P2' and x['concurrency']==1)
text += [f"P2 C1三轮均值：输出 **{r['output_tps']:.2f} tokens/s**；TPOT **{r['tpot_mean']*1000:.2f} ms/token**；单请求耗时 **{r['e2e_mean']:.2f} s**；TTFT **{r['ttft_mean']:.3f} s**。", '',
 '## Prefill、长上下文和延迟','',
 '下表均为三轮统计值的算术平均。P50/P95/P99是每轮分位数的平均，不冒充合并样本分位数。所有并发的完整延迟和请求/s在medium-aggregates.json。输入tokens/s用完整批次墙钟作分母，包含Decode，不能当纯Prefill算子吞吐。', '',
 '| 档位/C | 输入tokens/s | 输出tokens/s | 请求/s | TTFT P50/P95/P99(s) | TPOT均值(ms) |',
 '| --- | ---: | ---: | ---: | --- | ---: |']
for b in [3,4,5]:
    for c in [1,32,64]:
        r=next(x for x in aggregate if x['bucket']==f'P{b}' and x['concurrency']==c)
        if not r['success_repeats']:continue
        q=r['ttft_seconds']
        text.append(f"| P{b}/C{c} | {r['input_tps']:.2f} | {r['output_tps']:.2f} | {r['rps']:.4f} | {q['p50']:.3f}/{q['p95']:.3f}/{q['p99']:.3f} | {r['tpot_mean']*1000:.2f} |")
text+=['', '## 15分钟稳定性：P2 C64','']
st=s['stability'];rates=[w['stream_arrival_output_tokens_per_second'] for w in st['windows']]
text += [f"持续提交900秒，含排空总时长{st['elapsed_including_drain_seconds']:.2f}秒；完成{st['requests']}请求，失败{st['failures']}，抢占{preempt('P2'):.0f}。", 
 f"15个一分钟窗口按实际SSE Token到达计数：平均 **{statistics.mean(rates):.2f} tokens/s**，范围{min(rates):.2f}–{max(rates):.2f}，窗口变异系数 **{statistics.pstdev(rates)/statistics.mean(rates):.2%}**。含排空总吞吐{st['output_tokens_per_second_including_drain']:.2f} tokens/s。",
 f"全部完成请求TTFT P95/P99={st['ttft_seconds']['p95']:.3f}/{st['ttft_seconds']['p99']:.3f}s；TPOT P95/P99={st['tpot_seconds']['p95']*1000:.2f}/{st['tpot_seconds']['p99']*1000:.2f}ms/token。",
 '![稳定性窗口吞吐](stability-final.png)','',
 '## 显存、缓存和并发容量','',
 '独立内存RPC采集P2/P3低高并发以及P4/P5上下文。每卡参数32.30GiB，模型buffer0.125GiB，去重合计32.43GiB；缓存物理分配18.10GiB，916块。KV与DeltaNet状态按共享物理storage去重，不能把各逻辑视图再相加。图池约0.52GiB，日志峰值激活0.84GiB、non-torch0.44GiB；工作区未独立分离，不编造精确占用。',
 'P4/P5诊断torch allocator最大allocated约51.40GiB、reserved约51.95GiB；与NPU总HBM统计口径不同。以下为5秒采样HBM观测峰值，含该单元预热；不能视为瞬时峰值。完整每单元结果在memory-observed.json。', '',
 '| 档位 | 最大成功客户端并发（三轮共同） | 对应点实际运行请求数上限（5秒观测） | 卡6/7 HBM观测峰值(MiB) |',
 '| --- | ---: | ---: | --- |']
for b in range(1,6):
    high=32 if b==5 else 64
    peak={card:max((hbm[f'medium-r{r}-P{b}-C{c}'].get(card,0) for r in [1,2,3] for c in CONC),default=0) for card in ['6','7']}
    active=max(running.get(f'medium-r{r}-P{b}-C{high}',0) for r in [1,2,3])
    text.append(f"| P{b} | {high} | {active:.0f} | {peak['6']}/{peak['7']} |")
text+=['', 'P5 C64三轮均因600秒socket-read超时在预热结束；无正式吞吐结果，也无直接证据断言OOM。最大成功客户端并发32不等于同时32请求驻留或运行；实际运行数见监控快照。', '',
 '## Graph执行覆盖','',
 '配置FULL_AND_PIECEWISE、enforce_eager=False，捕获尺寸1/2/4/8/16/32/64；各服务日志观察到FULL、PIECEWISE、NONE路径。大Token prefill按已冻结覆盖规则走NONE，不声称全部图回放。Medium正式服务日志未匹配到显式recompile/fallback告警；这不能替代逐算子证明。请求block_size128经后端对齐后实际为2048，混合状态页有padding；所有配置按同一规则解释。', '',
 '## 独立Profiler','']
profile=ROOT/'profile-analysis.json'
if profile.exists():
    rows=json.loads(profile.read_text())
    text += ['P2/P3的C1和C64共4组，每组2个rank，均有离线解析结果。采集限24个引擎迭代，并非完整请求；Profiler耗时不能与正式吞吐混算。时间来自CANN step_trace_time.csv，单位ms，计算/通信可能重叠。', '',
      '| 档位/C | rank | 计算 | 通信 | 未重叠通信 | overlap | HCCL Wait/Sync操作累计 | Free（不归因为同步等待） |',
      '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    for r in rows:
        q=r['step_time_us'];w=r['communication_operation_sums_ms']
        text.append(f"| {r['case']} | {r['rank']} | {q['Computing']/1000:.2f} | {q['Communication']/1000:.2f} | {q['Communication(Not Overlapped)']/1000:.2f} | {q['Overlapped']/1000:.2f} | {w['Wait Time(ms)']:.2f}/{w['Synchronization Time(ms)']:.2f} | {q['Free']/1000:.2f} |")
    text+=['','Wait/Sync为communication.json逐HCCL操作的原生字段累计，可能互有包含，不能彼此相加或当作完整请求壁钟。解析器报告部分memory record不完整，原采集也有RECORD状态停止提示；这限制了截断边界的解释。P3 C64的Wait/Sync字段全为0且通信归入Idle，不能将其解释为真实无同步开销。完整请求的同步等待归因仍缺乏可靠独立数值，Free不冒充同步等待。EP关闭，专家负载偏斜不适用。']
else:text+=['原始四组双rank数据已采集，但尚无最终离线解析结果；计算/通信和同步等待不填推测值。']
text+=['', '## 质量与结论范围','',
 'Small：48/70=68.57%。Medium：1414/2406=58.77%，学科宏平均61.17%，无效输出1，服务失败0。Medium质量输出按原始ID从Large复用，无额外模型调用。固定零样本、关闭thinking、正常EOS、max_tokens16、严格单字母判分，不等同官方CoT成绩，也不能用不同样本集准确率比较性能配置退化。',
 '本次仅完成B0 TP2基线及预算筛选；尚未运行W1–W7权重布局候选，因此不能证明当前TP2是最优布局。短输入仍随并发增加而提高吞吐，长输入吞吐较早达到平台，P5还有排队/超时约束。后续权重布局实验默认Small+Medium，并沿用冻结预算和图覆盖规则作同条件对照。', '',
 '## 释放与证据','',
 '原实验结束时卡6/7无运行进程，HBM3433/3437MiB，已回到空闲基线。2026-10-26新SSH端口32037分配物理卡4/5，当前也无NPU进程、每卡3434MiB；这只是当前容器状态，不能将卡4/5身份回填历史测试。此次仅离线解析历史Profiler，没有重启模型或重跑基准。',
 '本地原始证据：../final-audit/final-evidence.json、npu-samples.jsonl、metrics-samples.jsonl、stability/P2/requests.jsonl、medium-campaign.log；完整请求与Profiler原始数据保存在服务器 /root/opset-research/b0-weight-layout-20261023/。',
 '数据hash、源码SHA、模型hash和来源映射在final-evidence.json的metadata中。不同规模不混合均值。']
(OUT/'FINAL_RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
fig,axes=plt.subplots(2,3,figsize=(14,8))
for b,ax in zip(range(1,6),axes.flat):
    rows=[r for r in aggregate if r['bucket']==f'P{b}' and r['success_repeats']]
    ax.errorbar([r['concurrency'] for r in rows],[r['output_tps'] for r in rows],yerr=[r['output_tps_sd'] for r in rows],marker='o',capsize=3)
    ax.set(title=f'P{b}: Medium, mean +/- SD (3 starts)',xlabel='Client concurrency',ylabel='Dual-NPU output tokens/s');ax.grid(alpha=.3)
axes.flat[-1].axis('off');fig.tight_layout();fig.savefig(OUT/'throughput-final.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,3.5));ax.plot(range(1,16),rates,marker='o');ax.set(xlabel='One-minute window',ylabel='SSE arrival output tokens/s',title='P2 C64: 15-minute continuous submission');ax.grid(alpha=.3);fig.tight_layout();fig.savefig(OUT/'stability-final.png',dpi=160);plt.close(fig)
print('Final report generated',OUT/'FINAL_RESULTS.md')
print('P2 C1',next(r for r in aggregate if r['bucket']=='P2' and r['concurrency']==1))
