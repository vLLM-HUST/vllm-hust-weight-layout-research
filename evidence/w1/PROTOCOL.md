# W1/P0 准备协议（2026-10-26）

状态：W1 Small已完成，12个性能单元/384条正式性能请求均成功，质量70题完成。相对历史B0总体吞吐比几何平均变化-1.80%，无明确收益；结束于Small，未执行Mini/Medium/Large。双卡显存已回到3434MiB/卡，无推理进程。完整结果见RESULTS.md。

## 假设与唯一变量

B0：TP2/DP1/PP1、EP off。W1：TP2/DP1/PP1、EP on。
目标：注意力继续 TP2，256 个路由专家预期每卡 128 个完整专家，专家本地中间维度预期从 256 变成 512。必须通过实际模型形状/专家映射确认，不只依据命令行。共享专家、路由器与通信实现按当前后端实测记录，不擅自推断。

## 冻结配置

Small：70 题质量、P1–P4 各 32 条性能请求；并发 1/8/32，12 性能单元/臂/轮。
BF16，max-num-seqs=32，max-num-batched-tokens=4096，显存利用率0.85。
Graph FULL_AND_PIECEWISE，enforce_eager=False；捕获1/2/4/8/16/32；图覆盖遵守原B0边界，较大prefill可能走NONE，必须披露。
MTP/APC/async关闭；chunked prefill开启；KV auto、请求block128（后端有效值需记录）、mamba-cache-mode none。
P1/P2上下文4096，P3为16384，P4为65536。种子20261023；沿用冻结客户端的Token ID、SSE、EOS、超时、预热与评分方式。
主模型路径 `/data/shared_upload/models/vspec/Qwen3.5-35B-A3B`。
vLLM HEAD `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`；Ascend HEAD `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`。每臂冻结实际dirty diff、包版本、模型和数据hash；差异需先解释。

## 当前硬件与基线

SSH端口32037，物理卡4/5，逻辑设备0/1，HCCS。空闲HBM3434MiB/卡。
CPU配额8核，主存限制64GiB。TP0绑定96–119，TP1绑定48–71，API24–31，Engine32–39，NUMA2/4交错，与旧B0策略一致。
旧B0物理卡为6/7，Small仅一轮。先直接与其同配置结果筛选，有收益再做当前卡Mini配对。Medium8192预算不能拿来计算Small4096预算的收益。

## 执行顺序

1. 确认双卡空闲、端口18124空闲；核验有效Graph配置和EP功能支持。
2. 首轮仅W1-r1完整Small，对比历史B0；出现启动/质量/数值异常先诊断，不进入Medium。
3. 有正向信号再做Mini B0-r1→W1-r1→W1-r2→B0-r2。Mini从Small固定种子顺序取P2/P3各16条，C1/C16；若收益主要来自P4，增加P4的16条C16。服务端仍为序列32、预算4096。
4. 每轮保存原始请求、失败、launch、源码/hash、图路径及释放后NPU状态；每臂使用独立编译缓存。
5. 无Small明确收益只在Small诊断。本目录无Medium/Large执行入口。

## 预先固定的晋级规则

初筛：报告12个Small单元相对历史B0的收益，不把跨时段差异直接作为最终结论。Mini主指标：4个同条件单元的W1/B0吞吐比等权几何平均，两组配对均>1，整体>=1.05。追加P4独立报告，不事后更换主指标。Mini仅支持这些负载的收益确认。
所有单元须正式完成，输出Token数一致，无新增请求错误/非有限值；质量不低于同轮B0且无新增无效回答。记录题ID差异；70题只是筛选，不能证明统计等价。
作为暂定保守延迟护栏，每个Mini单元两轮均值TTFT P95与TPOT P95相对B0不超过1.10；任何单元平均吞吐回退超过5%时不自动晋级。此门槛是实验选择规则，不声称统计显著性。
仅部分单元获益时，明确标为特定负载候选，在Small复测并另行冻结范围；不能事后删负向单元以满足总体门槛。
通过后才准备Medium-v2同条件B0/W1对照，保留8192预算/序列64及正式三轮规则；不自动开始Large。

## 启动方式（准备阶段不执行）

在服务器source `/root/opset-research/env.sh` 后：

```bash
cd /root/opset-research/w1-weight-layout-20261026
/root/opset-research/.venv/bin/python run_small.py --arm W1 --repeat 1
# 上行只输出计划；实际执行才附加 --execute
```

`client_snapshot.py` 来自原B0客户端，只调整输出根、端口、EP开关及当前物理卡标注，移除旧main避免误入Large队列。服务名称仍保留b0-qwen35以保持请求接口一致，实验臂以独立目录与launch EP参数识别。
不调用原run_campaign.sh或原run_b0.py的main。运行结束只停止本次创建的服务进程组；检查npu.final.txt无本次残留且回到空闲显存。
当前准备脚本复用原客户端的batch结果结构：预热失败的failures字段是旧格式，不得将其当作正式请求失败数；依据status与warmup.json独立统计。
