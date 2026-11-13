"""Render the already-copied read-only snapshot; never touch remote experiment."""
import datetime,json
from pathlib import Path
root=Path(__file__).resolve().parent
s=json.loads((root/'progress-snapshot.json').read_text(encoding='utf-8-sig'))
cells=s['cells']
stamp=(datetime.datetime.fromisoformat(s['utc'])+datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
text=['# B0 阶段实验报告（未完成）','',f'快照时间：北京时间 {stamp}。', '',
 '模型 Qwen3.5-35B-A3B，双 Ascend 910B2，BF16，TP2/DP1/PP1，EP 关闭。',
 'Medium-v2：性能每档64条，质量2406题；并发1/4/8/16/32/64；序列预算64，token预算8192；三轮独立启动。', '',
 f"已有结果单元 {s['completed_medium_cells']}/90（{s['completed_medium_cells']/90:.1%}），状态 {s['medium_status_counts']}。这表示单元进度，不表示耗时进度。",
 f"各轮已有结果：{s['by_repeat']}。当前单元：{s['status'].get('cell')}。", '',
 'Small 已完成并复用；Medium 质量按ID复用 Large 输出。Large 的批次吞吐保留原标签，不混入 Medium 平均值。',
 '稳定性、四组Profiler及长上下文显存诊断尚未完成，因此不能报告最终尾延迟、通信占比或三轮稳定收益。', '',
 '## Medium 第一轮双卡总输出吞吐', '',
 '以下为单轮实测，单位 tokens/s；不得视为三轮平均或持续满载吞吐。', '',
 '| 输入/输出 tokens | C1 | C4 | C8 | C16 | C32 | C64 |',
 '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for b,label in [(1,'512/128'),(2,'1024/1024'),(3,'8192/256'),(4,'32768/256'),(5,'131072/256')]:
    vals=[]
    for c in [1,4,8,16,32,64]:
        r=cells[f'medium-r1-P{b}-C{c}'];v=r.get('dual_card_output_tokens_per_second')
        vals.append(f'{v:.2f}' if v is not None else r['status'])
    text.append('| '+label+' | '+' | '.join(vals)+' |')
r=cells['medium-r1-P2-C1']
text+=['','## 低并发 Decode（Medium 第一轮 P2 C1）','',
 f"双卡输出 {r['dual_card_output_tokens_per_second']:.2f} tokens/s；平均 TPOT {r['tpot_seconds']['mean']*1000:.2f} ms/token；平均单请求耗时 {r['e2e_seconds']['mean']:.2f} s；平均 TTFT {r['ttft_seconds']['mean']:.3f} s。", '',
 '## Prefill 与长上下文（第一轮）', '',
 '| 档位/并发 | 输入 tokens/s | TTFT P50/P95/P99（s） | TPOT 均值（ms） |',
 '| --- | ---: | --- | ---: |']
for b in [3,4,5]:
    for c in [1,64]:
        r=cells[f'medium-r1-P{b}-C{c}']
        if r['status']!='completed':continue
        t=r['ttft_seconds']
        text.append(f"| P{b}/C{c} | {r['input_tokens_per_second']:.2f} | {t['p50']:.3f}/{t['p95']:.3f}/{t['p99']:.3f} | {r['tpot_seconds']['mean']*1000:.2f} |")
text+=['','输入吞吐的分母为完整请求批次墙钟，包含输出阶段，不是纯 Prefill kernel 吞吐。', '',
 '## 质量', '', '| 数据集 | 正确/总题数 | 准确率 | 无效输出（含失败） |', '| --- | ---: | ---: | ---: |']
for name in ['small-quality','medium-quality']:
    q=cells[name]['quality']
    text.append(f"| {name} | {q['correct']}/{q['total']} | {q['accuracy']:.2%} | {q['invalid_including_failed']} |")
text+=['','质量为固定模板、关闭 thinking、严格单字母判分的结果，不等同官方推理/CoT成绩。Small 与 Medium 样本组成不同，不能用准确率差异判断性能配置导致退化。', '',
 '## 失败点与显存', '',
 'P5 C64 第一、二轮均在预热出现 SocketTimeoutError（600秒 socket-read deadline），正式计时未开始。原 summary 的 failures=64 是计划请求数量，不能解释成64次正式请求失败。尚无证据把它确定为 OOM。',
 '当前记录的长上下文最大成功客户端并发为 P5 C32；实际同时运行请求数需结合调度指标，不能与客户端并发混同。', '',
 '第一轮4096上下文服务日志：每卡权重32.43 GiB、KV/状态预算18.11 GiB、图内存0.52 GiB、峰值激活0.84 GiB、non-torch内存0.44 GiB。项目可能来自不同统计口径，不直接相加为同时驻留峰值。',
 '本次核查时 npu-smi：卡6/7健康，HBM57349/57351 MiB；这是当前占用，不是整个实验的峰值。',
 '实验仍运行，因此未释放正在使用的显存。完成后仅停止本实验所属进程并核验回到空闲基线。', '',
 '原始快照：progress-snapshot.json。所有数据来自已保存的 summary；本报告没有打断服务或发起额外推理。']
out=root/'report';out.mkdir(exist_ok=True)
(out/'PROGRESS.md').write_text('\n'.join(text),encoding='utf-8')
print(out/'PROGRESS.md')
