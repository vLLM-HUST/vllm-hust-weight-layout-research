# 实验结果与数据索引

所有路径相对于仓库根目录；复制文件的来源及SHA-256见 `provenance.json`。

| 编号 | 实验结论 | 优先核查的证据 | 边界 |
|---|---|---|---|
| F1 | 融合适配、舍入语义修复 | `evidence/fusion/README.md`；`raw/data/artifacts/matched-source-20261014/`；`raw/data/results/shared-rounded-validation.json`、`fusion-poison-probe.json` | 固定环境、FP16/BF16、小M等已验证条件 |
| F2 | 短/长TPOT改善1.97%/0.67% | fusion下六个`20261014T*-baseline-r*`/`fusion-r*`目录中的`measurements.json`、`manifest.json`、`quality.json`；`matched-raw-metrics.csv` | 三个生命周期；配对区间包含零；C2异常结果不作提速主张 |
| F3 | zeros→empty禁用 | `production-shape-probe-m20.json`、`output-poison-probe-ordered.json`、对应日志及`repair-conclusion.json` | 正确性反例；根因未完全定位 |
| G1 | 兼容任务切换可行、原生链替换失败 | `evidence/graph-switch/README.md`和`raw/route-b-20261015/` | 单卡算子探针；没有端到端性能结论 |
| B1 | B0 Medium 87成功/3超时、P2 C64 944.92 | `evidence/b0/final-audit/final-evidence.json`中的cells；`medium-campaign.log`；`report/medium-aggregates.json` | 目前能从采集摘要复算；完整Medium逐请求文件待补 |
| B2 | B0质量1414/2406 | `evidence/b0/cells/large-quality/requests.jsonl`；`evidence/dataset/medium/manifest.json`与`quality.answers.jsonl.gz` | 按ID抽取复用；零样本、关闭thinking、max_tokens16，不等同官方CoT成绩 |
| B3 | 15分钟稳定性无失败、896请求 | `evidence/b0/final-audit/stability/P2/requests.jsonl`及final-evidence的stability字段 | 持续提交900秒，另有排空；窗口均值与全程吞吐分开 |
| W1 | Small 12单元、384条、比值几何平均−1.80% | `evidence/w1/w1-small-r1/cells/*/requests.jsonl`、summary；`evidence/b0/cells/small-*/` | 单轮，B0物理卡6/7、W1物理卡4/5；历史筛选，非配对因果结论 |
| W2 | EP开启、未进入Medium、显存释放 | `evidence/w1/w1-small-r1/servers/`、`execution.complete.json`、`RESULTS.md` | 准备的run_mini脚本不代表已执行；释放是历史记录，不是本次实时核查 |

完整融合Profiler原始文件在原始归档中，Git中不重复放入大型trace/DB。保留早期失败记录；不能只抽取表现最好的轮次。
