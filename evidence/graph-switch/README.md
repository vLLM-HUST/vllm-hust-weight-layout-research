# Route B：NPUGraph 局部任务组更新最小探针

日期：2026-10-15（Asia/Shanghai）。不启动模型，不修改 OPset 或引擎生产代码。

## 结论

路线 B **部分可行**：同一张 NPUGraph 内，兼容的单内核方案可以更新并切回；当前环境不能直接把原生 Persistent MatMul + SwiGLU 任务链改为单个融合内核。不能把局部参数更新能力等同于任意子图拓扑替换能力。

| 探针 | 结果 | 有效检查 |
|---|---|---:|
| 单 Add 任务，标量 2/3 往返 | 通过 | 15 |
| 单任务 Add/Mul 往返，使用各自 oracle | 通过 | 15 |
| 两个算术任务改为一个任务 | 拒绝：507000，total=2/success=1/failed=1 | 原结构 3 次通过，切换未成功 |
| Persistent MatMul + SwiGLU → 融合内核（v2） | 拒绝：kernel type error，retCode=0x7080005 | 原结构 3 次通过，切换未成功 |
| 融合 Triton 内核 BLOCK_N=128/64 往返 | 通过 | 15 |

每个成功实验只捕获 1 次，之后执行 A→B→A→B→A，每个阶段 3 次 replay。改变输入、输出填 NaN，并检查组外公共后缀 `z=y+1`。更新使用独立 stream 和 ExternalEvent，先等旧工作完成，不尝试并发更新。

Triton 探针是同一融合算法的两种分块特化，不是两个独立算法。为区别是否真正更新，B 同时切换到另一组权重（-W），oracle 使用对应权重；这是诊断手段，不是模型权重热更新功能。它验证兼容单任务切换，不证明任意 Triton 方案可互换。

矩阵探针：BF16，M=7、激活输出 N=256、K=2048，权重形状 [512,2048]。参考为 materialized `F.linear` + native `npu_swiglu`，容差 rtol=.03/atol=.05。未跑多 shape、TP、完整模型、并发、取消或服务端动态配置测试。

## 原生路径适配与失败记录

原生算子返回新张量，需把结果写到固定边界输出。初版使用 copy_，在 task group 捕获阶段遇到 DMA memcpy 不支持（507009），属于探针适配限制，不能据此否定融合更新。

v2 将适配改为 `torch.add(act,0,out=y)`。捕获成功，原生任务组自更新及 replay 正确，随后替换为融合内核时出现 kernel type error。这里存在多项变化（任务数量、任务类型），因此不能把最终失败唯一归因于数量，或宣称所有任务类型转换都不支持。简单的 Add/Mul 单任务转换实际通过。

两任务→一任务的算术诊断使用不同表达式、各自 oracle，意在隔离任务数量限制，不是等价融合性能实验。原生链包括保留的清零和固定输出适配，不把它描述成恰好两个设备任务。

## 环境

- Ascend 910B2，使用容器逻辑 NPU 0（物理 NPU 1）；两卡起止均无其他计算进程。
- torch 2.10.0+cpu、torch_npu 2.10.0.post4、triton-ascend 3.2.2。
- CANN 9.1.0；driver 26.0.rc1。
- Ascend commit：5cb98caaadeff42b5b62b996e34bb2aaa29d20fd。
- vLLM commit：0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665。

这是固定版本线的负/正结果，不外推为最新软件栈的一般限制。

## 复现与证据

原始 JSON、日志、脚本、环境回执和相对路径 SHA-256 manifest 位于同目录的 `raw/route-b-20261015/`。环境回执含原始运行路径，仅作 provenance；manifest 不依赖这些路径。

在已配置的研究环境目录中运行：

```bash
source env.sh
python route_b_probe.py --mode control --output control.json
python route_b_probe.py --mode kernel --output kernel.json
python route_b_probe.py --mode topology --output topology.json
python route_b_probe.py --mode fusion --output fusion.json
python route_b_probe.py --mode triton --output triton.json
```

预期 topology/fusion 失败并退出；不要在失败的 Graph 上继续 replay。每条命令使用独立 Python 进程。归档脚本为最后版本；初版 fusion.log 对应唯一差别为 `torch.add(act,0,out=y)` 之前的 `y.copy_(act)`。

`host_update_sync_ms` 只保留为探针诊断，不是稳态性能基准或服务收益，不能据此报告 TPOT/token/s 提升。本轮没有修改插件、提交 PR 或发布 PyPI。

## 下一步

若优先做路线 B，应先限制为兼容单任务方案（例如同一融合算子的分块特化），补齐固定输出接口和方案契约。若目标仍是原生链↔融合链，当前探针未通过，应研究任务类型/拓扑兼容性或转向局部双子图，不直接接入服务。
