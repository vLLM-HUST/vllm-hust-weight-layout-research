# OPset 修复与性能验证：2026-10-14

## 结论

- **共享专家 Linear–SwiGLU：适配和数值修正通过。** 已接入本地及服务器开发插件，实际插件入口完成 TP2 Graph 推理，出现 80 条共享专家命中日志。未发布 PyPI、未提交 PR。
- **Persistent MatMul 的单纯 zeros→empty：当前环境 No-Go。** 实际模型、独立随机输入和非零/NaN 输出预填充均提供反例。当前失败后端组合会报告 `blocked` 并保留原函数及 zeros。未找到能通过完整门禁的安全替代，不宣称其在所有后端不可修复。
- **端到端收益较小，尚不能证明稳定提速。** 单请求短/长 prompt 的 TPOT 均值下降 1.97% / 0.67%，三轮配对差异区间包含零。正、零、负结果均保留。

## 改动与原因

Qwen3.5 MoE 的共享专家位于 `moe_forward_shared` 自定义算子内部，外层 FX pass 看不到 Linear→SwiGLU。新增内部 split-method 适配，保留调用方 stream/event 边界、down projection 与 expert gate。

原融合内核直接对 FP32 GEMM 累加值做激活，跳过了原路径的 BF16/FP16 舍入边界。修正为先按输入 dtype 舍入 gate/up，再做激活。当前内部路径仅覆盖 batch-invariant NPU、FP16/BF16、小 M（1–128）、无 bias/量化/LoRA/权重预取的已知结构；其他情况保持原路径，未知 split-method 结构拒绝安装。

`empty` 的首个实际反例是输入 `(20,2048)`、权重 `(6144,2048)`：输入/权重有限，empty 输出非有限，同输入 zeros 输出有限。这证明当前后端下“已有输出内容不影响结果”的前提不成立，但尚未确定具体 compiler lowering 根因。

## 验证覆盖

| 验证 | 结果与范围 |
|---|---|
| 融合 changing-input Graph replay | FP16/BF16，14 个 dtype/shape 组合 ×3，42 次比较通过；测试输出逐元素一致 |
| 正式融合内核输出预填充 | 非零 7 / NaN、实际及 tail shape，16 组合 ×3，48 次原生参考、有限值和同输入确定性检查通过 |
| 模型 matched | 3 对独立 baseline/fusion 服务；三轮 Transformer smoke 均完全一致 |
| 质量回归样本 | 固定前 20 条 MMLU-Pro；每臂重复三轮，每轮 13/20=65%，三轮各 20/20 预测一致，请求完全一致 |
| 插件测试 | 11/11 单元测试通过，包含失败后端保留原函数、未知共享专家结构拒绝 |
| 安全检查 | 两张卡 Tensor 创建、同步、回 CPU 通过；所有本次 vLLM/EngineCore/诊断进程已退出 |

MMLU-Pro 只有 **20 个独立样本**，不是 120 个独立题目，也不是完整官方评测。性能 workload 是自定义 Transformer prompt，不是 MMLU-Pro 任务吞吐。其他四个主数据集及完整五数据集合同未在本轮执行。

## 三轮性能结果

Qwen3.5-35B-A3B、BF16、双 910B2、TP2、batch-invariant、Graph；同一模型/运行时/数据顺序/配置。短输入 20 token，长输入 939 token，均生成 64 token。

每个服务生命周期、每种 C1 workload：3 次预热，5 次计时；每臂每种 workload 共 15 个 C1 样本。均值与 SD/方差基于 **3 个生命周期均值**，不把同一进程的请求当作独立服务重复。启动、权重加载、编译、capture、预热、profiling 均不计入下表。

| 单请求 workload | 指标 | B0 均值 ± SD | B1 均值 ± SD | 均值改善 |
|---|---|---:|---:|---:|
| 短 | TPOT ms | 15.1753 ±0.6060 | 14.8757 ±0.2882 | 1.97% |
| 短 | output token/s | 45.2799 ±1.6450 | 46.0448 ±0.6989 | 1.69% |
| 短 | TTFT ms | 458.9163 ±14.0228 | 453.2137 ±7.8417 | 1.24% |
| 长 | TPOT ms | 15.0076 ±0.4393 | 14.9066 ±0.1949 | 0.67% |
| 长 | output token/s | 45.5523 ±1.3122 | 45.6304 ±0.5731 | 0.17% |
| 长 | TTFT ms | 460.4347 ±13.2165 | 463.8322 ±7.7877 | −0.74% |

配对 TPOT 差异（B0−B1）均值及描述性 95% t 区间：

- 短：0.2996 ms，区间 `[−0.4903, 1.0894]` ms。
- 长：0.1010 ms，区间 `[−0.5358, 0.7379]` ms。

只有三个生命周期，区间包含零，不据此宣称稳定总体收益。raw summary 还保留每轮、每请求的值、SD 与方差。

**并发 2 数据仅为 exploratory：**初始 runner 只独立预热了 C1；第一轮短 workload 的 C2 基线存在大幅波动。表观吞吐 +10.76%、TPOT −30.91% 不属于可靠结论，未删掉异常样本来制造提升。后续如研究 C2，需重新预注册并按 C2 分别预热复测。

## 算子级计时与 profiling

预加载常驻 NPU 张量，无模型加载和通信；每变体 5 次预热、30 个交替顺序样本，每样本同步后平均 20 次 Graph replay。此计时包含 Graph replay 的 host launch，不等同于纯 device compute。

BF16 `(M,N,K)` 中 N 是 SwiGLU 输出宽度；gate/up 权重为 `(2N,K)`：

| M,N,K | 图内算子组延迟下降 |
|---|---:|
| 1,256,2048 | 5.78% |
| 2,256,2048 | 6.25% |
| 4,256,2048 | 27.02% |
| 7,256,2048 | 3.18% |
| 32,256,2048 | 19.67% |
| 128,256,2048 | 15.08% |
| 7,513,257（tail） | 8.12% |

最后一对模型服务另外采集同一 32-token 请求的 NPU profiling，离线导出 CSV 与 MindStudio DB。每个 rank：

| kernel 类型 | B0 调用数 | B1 调用数 |
|---|---:|---:|
| linear_persistent | 8,672 | 7,392 |
| ZerosLike | 9,722 | 8,442 |
| SwiGlu | 2,560 | 1,280 |
| 新 shared_linear_swiglu | 0 | 1,280 |

这证明每 rank 的 1,280 组共享专家清零/投影/激活被一个融合内核替代。并不是全部 MatMul 或 routed experts 都被融合。

新融合内核占 B1 kernel 累计持续时间约 3.84% /4.11%（两 rank）。其余仍有 GroupedMatmul、线性投影、通信/同步等成本；因此单算子收益不会等比例转化为总体吞吐。kernel 时间可以重叠，通信 kernel 持续时间可能包含等待，**这些百分比不是端到端墙钟占比**。

## 保留的失败证据与停止门

| 尝试 | 结论 |
|---|---|
| 旧 empty plugin、独立 source rewrite、完整模型 eager | 均出现重复感叹号 |
| 原先 zeros-first、静态合成探针 | 曾通过，但未覆盖 empty-first/预填充依赖，不能用于恢复优化 |
| 真实 first-layer 观察 | inputs/weights finite，empty output nonfinite，zeros reference finite |
| inference mode /参数 Tensor 子类分离 | 普通 Tensor、参数子类、两种 inference 状态均可出现 NaN |
| 非零/NaN 输出预填充 | 未通过，结果依赖输出初始内容 |
| tile 对齐、显式 dot accumulator、ordered raw stream | 未关闭 poisoned-output 门 |
| 内核内清零 | 小矩阵 9 组通过；尝试完整 36-case dtype/tail 门时提前失败 |
| 内核内清零 +tile 对齐 | 数值不符，随后出现 MTE/GM 地址错误；停止候选，未部署 |

没有把小矩阵成功外推为完整模型安全，也没有对失败候选跑或报告性能收益。当前保留原 zeros，已知失败组合由插件 fail closed。进一步恢复该优化需要针对 backend/compiler 的修复及完整新证据，不属于本轮已解决项。

## 可复现记录

运行时固定：vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`；Ascend `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`。Torch 2.10.0+cpu、torch_npu 2.10.0.post4、triton-ascend 3.2.2、CANN 9.1.0；模型/数据 receipt 与脚本 hash 在原始归档内。不把本环境称为最新 main。

| 轮次 | B0 run ID | B1 run ID |
|---|---|---|
| 1 | 20261014T093553Z-baseline-r1 | 20261014T094345Z-fusion-r1 |
| 2 | 20261014T095342Z-baseline-r2 | 20261014T094906Z-fusion-r2 |
| 3 | 20261014T095813Z-baseline-r3 | 20261014T100333Z-fusion-r3 |

[完整原始归档](raw/opset-repair-evidence-20261014.tar.gz)，包含成功/失败日志、原始请求/输出、算子计时样本、两套 profiling CSV/DB、实际实验源码和修复插件。归档 SHA-256：

```text
91a8967a208da284d13ce4c749332f238f3e58e3679ffe64299889c283eaab8d
```

解压后的 `results/repair-manifest.json` 使用相对路径、size 和 SHA-256 建立索引。`artifacts/matched-source-20261014` 保存实际性能实验实现，避免后来安全保护修改混入原始实验。MindStudio 导入第三轮各 `profile/.../ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_*.db`，每臂两个 rank 的数据库。

本地修复源码和服务器开发源码已更新；公开 PyPI 和 GitHub PR 未改变。

中间诊断文件的复用名称及 tokenizer 解析纠正见 [artifact 说明](raw/diagnostic-file-notes.json)；以对应原始日志和纠正后的探针为准，不将文件名当作 variant 身份。
