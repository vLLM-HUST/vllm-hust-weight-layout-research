import json
from pathlib import Path

from compare_small import main as compare

ROOT = Path('/root/opset-research/w1-weight-layout-20261026')


def main():
    compare()
    d = json.loads((ROOT/'small-comparison.json').read_text())
    rows = d['cells']
    complete = d['completed_performance_cells'] == 12
    q = d.get('W1_quality', {})
    lines = ['# W1 TP2 + EP2：Small 实验结果', '',
        f'性能完成单元：{d["completed_performance_cells"]}/12。', '',
        '| 单元 | 历史 B0 tokens/s | W1 tokens/s | 变化 |',
        '| --- | ---: | ---: | ---: |']
    for r in rows:
        if 'ratio' in r:
            lines.append(f'| {r["name"]} | {r["b0_tokens_s"]:.2f} | {r["w1_tokens_s"]:.2f} | {r["change_pct"]:+.2f}% |')
        else:
            lines.append(f'| {r["name"]} | — | — | {r["status"]} |')
    if d['geomean_change_pct'] is not None:
        lines += ['', f'已完成单元吞吐比等权几何平均变化：{d["geomean_change_pct"]:+.2f}%。'
                  + ('完整12单元。' if complete else '尚未完成，不作最终结论。')]
    lines += ['', f'W1质量：{q.get("correct")}/{q.get("total")}；'
              f'无效/失败：{q.get("invalid_including_failed")}。历史B0为48/70。',
              '', '答案变化：', '']
    for r in d.get('quality_changes', []):
        lines.append(f'- {r["id"]}：B0={r["b0"]!r}，W1={r["w1"]!r}。')
    if complete:
        change = d['geomean_change_pct']
        if change < 5:
            lines += ['', '## 阶段决策', '',
                f'12单元总体吞吐变化{change:+.2f}%，没有达到预先约定的明确收益门槛。',
                '本阶段结束于Small；未运行Mini配对或Medium/Large。',
                '结果支持当前EP开关方案没有显示足够收益，不证明所有EP实现或参数均无收益。',
                '未作收益性结论，因此没有追加正式Profiler与15分钟Medium稳定性。']
    baseline_env = Path(d['baseline'])/'environment.json'
    candidate_env = Path(d['candidate'])/'environment.json'
    if baseline_env.exists() and candidate_env.exists():
        a, b = json.loads(baseline_env.read_text()), json.loads(candidate_env.read_text())
        parity = {'model_hashes_equal': a['model_files'] == b['model_files'],
                  'dataset_manifest_equal': a['dataset_manifest_sha256'] == b['dataset_manifest_sha256'],
                  'packages_equal': a['packages'] == b['packages'],
                  'repos': {k: {'head_equal': a['repos'][k]['head']['stdout'] == b['repos'][k]['head']['stdout'],
                                'diff_equal': a['repos'][k]['diff_sha256'] == b['repos'][k]['diff_sha256']}
                            for k in a['repos']}}
        (ROOT/'environment-comparison.json').write_text(json.dumps(parity, indent=2))
    lines += ['', '## 实验配置与证据边界', '',
        'Small 70题质量，性能P1–P4各32条，C1/C8/C32；BF16、TP2/DP1/PP1、EP开启。',
        '序列32、Token调度预算4096、显存利用率0.85、MTP/APC/异步关闭。',
        'FULL_AND_PIECEWISE，捕获1/2/4/8/16/32；日志观察FULL回放，大Token prefill保留后端NONE路径。',
        '当前物理卡4/5；历史B0为6/7，同型号910B2、HCCS。历史Small只有一轮，跨时段对比用于筛选。',
        '启动日志记录每rank128个专家、EP组开启，每卡权重加载32.4253 GB。',
        '请求、预热、launch、源码与权重/数据hash、服务日志和释放记录保存在w1-small-r1/。',
        'Mini数据计划从Small P2/P3各取16条，C1/C16；仅有正向信号时执行。',
        'Medium只有Small收益经复测明确后执行；Large默认不运行。', '']
    final_npu = ROOT/'w1-small-r1/npu.final.txt'
    if final_npu.exists():
        state = final_npu.read_text()
        lines += ['## 最终NPU状态', '', '```text', state.strip(), '```', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
