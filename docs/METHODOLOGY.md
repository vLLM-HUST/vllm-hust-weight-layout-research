# 数据完整性与复核方法

## 数据覆盖

融合三轮baseline/fusion的请求与输出、质量记录、启动manifest、日志、实际实验源码快照、诊断反例，以及包含Profiler的完整原始压缩包；Graph探针原始数据；B0 Small逐请求与Medium采集摘要、稳定性和监控；W1 Small全部请求、配置与日志。数据集保留Small、medium-v2及父manifest；Large仅保留用于质量复用的输出，不混入Medium性能统计。

离线脚本 `tools/verify_evidence.py` 核对已复制文件SHA-256，重新计算融合C1的三轮TPOT、W1/B0吞吐比几何平均、B0 Medium单元成功/失败数量和P2 C64均值，以及从Large输出按ID提取的Medium质量分数。结果写入 `verification/results.json`。

SHA-256用于核对此次复制前后内容一致。源文件中的历史manifest、日志和时间信息原样保留。

## 尚待补齐

1. B0 Medium三轮正式性能单元的完整`requests.jsonl`、预热记录，以及完整原始Profiler文件。本地`final-evidence.json`包含采集摘要与部分分析数据，但不能替代这些原始文件。历史报告记录远端目录 `/root/opset-research/b0-weight-layout-20261023/`；应从仍可访问的原容器/备份补取，再与已有配置、单元摘要及hash核对。本次未启动远端服务或补跑实验。

## 统计口径与结论范围

- 融合有算子级改善，但端到端短/长C1的小幅均值改善未证实稳定；C2未充分单独预热，保留数据但不将其大幅表观改善作为可靠性能结论。
- 融合中的B0/B1并非权重分配B0/W1。
- Small使用序列预算32、token预算4096；Medium使用序列预算64、token预算8192。不能跨规模直接算优化增益。
- W1对照历史B0，设备编号和时段不同；未做Mini配对。−1.80%是描述性筛选结果。
- P5 C64三次为预热读取超时，无正式吞吐，不能填0，不能无证据判定OOM。
- 本次材料整理未改变任何实验数据；没有运行Medium/Large队列，也没有占用实验显存。
