# MMLU-Pro 权重分配实验集 v1

本数据集仅使用服务器已有 MMLU-Pro。真实质量测试与合成性能测试分离；不包含 BurstGPT、SyFi 或外部新增语料。

## 数据清单

| 版本 | 质量题目 | 固定长度性能请求 |
| --- | ---: | ---: |
| Small | validation 全部 70 题，14 学科各 5 题 | 128 条，P1–P4 各 32 条 |
| Large | test 全部 12,032 题，保留原始学科比例 | 1,280 条，P1–P5 各 256 条 |

每个版本包含 `quality.requests.jsonl.gz`、`quality.answers.jsonl.gz`、`performance.requests.jsonl.gz`。答案文件只交给评分端，不传给模型。请求中不含原始答案或 `cot_content`。

| 档位 | 实际输入 Token | 固定输出 Token | 所需最小上下文 |
| --- | ---: | ---: | ---: |
| P1 | 512 | 128 | 640 |
| P2 | 1,024 | 1,024 | 2,048 |
| P3 | 8,192 | 256 | 8,448 |
| P4 | 32,768 | 256 | 33,024 |
| P5 | 131,072 | 256 | 131,328 |

## 输入契约

Token 化使用 `/data/shared_upload/models/vspec/Qwen3.5-35B-A3B` 的本地 tokenizer 和 chat template，关闭 thinking。文件 SHA256 见 manifest。性能请求的 `prompt_token_ids` 是权威输入：通过支持 Token ID 数组的 completions 接口或离线推理接口提交，不再套 chat template，也不要先 decode 再 encode。质量请求同时提供 messages 和 Token IDs；严格对比优先使用已冻结的 IDs。

性能正文由 validation 的题干和选项随机排列、循环拼接并按 Token 预算截断构成；保留完整外层 chat 控制标记。它不是一道完整的官方试题，也不用于质量评分。不同记录使用不同种子和编号，Small/Large 请求不同，但语料都来自相同 70 道验证题，不能视为语义独立的真实内容留出集。长文本包含重复片段，因此主对照关闭 prefix caching。

Large test 不用于构造性能文本或拟合专家放置。Small 可用于开发；Large 质量题用于冻结候选后的正式验证。性能集不足以支持对真实代码等其他领域专家路由的泛化结论。

## 推理与评分

- 质量：temperature=0，seed=20261023，max_tokens=16，正常 EOS 停止；去除首尾空白后只接受一个合法大写选项字母，其余记 invalid，并按错误计入准确率。分母包括失败和无效回答，不悄悄重试或排除。
- 质量输出：总体准确率、14 学科准确率、宏平均、invalid/失败数，以及与同环境 B0 的答案变化。单次 greedy Token 差异不自动判为数值错误，应另做定位。
- 性能：temperature=0，seed=20261023，ignore_eos=true，max_tokens=每条 output_tokens。校验实际生成长度，失败/短输出独立报告，不能当作吞吐收益。
- 主对照：BF16、MTP 关闭、prefix caching 关闭；模型、tokenizer、上下文上限、调度预算、输出规则在比较单元内一致。当前 4096 上下文配置不能运行 P3–P5。
- Small：先质量与布局检查，再 P1–P4 并发 1/8/32。Large：质量全量一次，性能 P1–P5 并发 1/4/8/16/32/64，至少三个独立启动的重复。资源不足的单元记录容量失败，不临时缩短输入。
- 每个性能单元处理整个对应桶一次；桶内顺序用相同固定种子打乱；跨方案使用相同顺序。预热独立、不计入测量；短单元若需延长，按完整固定轮次重复并记录轮次，不能选择性重放。
- 报告双卡总输出 tokens/s、TTFT、TPOT、P95/P99、显存与失败率。各长度档独立报告，不用单一混合平均掩盖退化。短样本尾延迟仅作为描述性结果。

## 重建与完整性

`build_dataset.py` 是服务器重建脚本，使用实验环境 `.venv/bin/python`；输出目录已存在时拒绝覆盖。源 Parquet、tokenizer 文件及所有生成压缩文件的 SHA256、数量和校验结果见 `manifest.json`。

脚本验证所有 gzip/JSON 记录可读、请求 ID 唯一、Token 数精确、性能输入 hash 一致，以及性能来源仅为 validation。gzip 时间戳可能使不同重建的压缩文件 hash 不同；性能记录 Token hash 和构建种子用于比对逻辑内容。

服务器目录：`/root/opset-research/datasets/mmlu-weight-layout-v1/`。
本次仅生成数据，不启动模型推理，不构成性能或质量实验结果。
