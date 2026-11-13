# B0 实验结果

已记录单元状态：`{'completed': 7}`。所有数值来自原始结果；未完成单元不填推测值。

| 档位 | 并发 | 成功重复/已测重复 | 双卡输出 tokens/s 均值 | 标准差 |
| --- | ---: | ---: | ---: | ---: |
| P1 | 1 | 0/0 | 未取得成功结果 | — |
| P1 | 4 | 0/0 | 未取得成功结果 | — |
| P1 | 8 | 0/0 | 未取得成功结果 | — |
| P1 | 16 | 0/0 | 未取得成功结果 | — |
| P1 | 32 | 0/0 | 未取得成功结果 | — |
| P1 | 64 | 0/0 | 未取得成功结果 | — |
| P2 | 1 | 0/0 | 未取得成功结果 | — |
| P2 | 4 | 0/0 | 未取得成功结果 | — |
| P2 | 8 | 0/0 | 未取得成功结果 | — |
| P2 | 16 | 0/0 | 未取得成功结果 | — |
| P2 | 32 | 0/0 | 未取得成功结果 | — |
| P2 | 64 | 0/0 | 未取得成功结果 | — |
| P3 | 1 | 0/0 | 未取得成功结果 | — |
| P3 | 4 | 0/0 | 未取得成功结果 | — |
| P3 | 8 | 0/0 | 未取得成功结果 | — |
| P3 | 16 | 0/0 | 未取得成功结果 | — |
| P3 | 32 | 0/0 | 未取得成功结果 | — |
| P3 | 64 | 0/0 | 未取得成功结果 | — |
| P4 | 1 | 0/0 | 未取得成功结果 | — |
| P4 | 4 | 0/0 | 未取得成功结果 | — |
| P4 | 8 | 0/0 | 未取得成功结果 | — |
| P4 | 16 | 0/0 | 未取得成功结果 | — |
| P4 | 32 | 0/0 | 未取得成功结果 | — |
| P4 | 64 | 0/0 | 未取得成功结果 | — |
| P5 | 1 | 0/0 | 未取得成功结果 | — |
| P5 | 4 | 0/0 | 未取得成功结果 | — |
| P5 | 8 | 0/0 | 未取得成功结果 | — |
| P5 | 16 | 0/0 | 未取得成功结果 | — |
| P5 | 32 | 0/0 | 未取得成功结果 | — |
| P5 | 64 | 0/0 | 未取得成功结果 | — |

## small-quality

```json
{
  "correct": 48,
  "total": 70,
  "accuracy": 0.6857142857142857,
  "invalid_including_failed": 0,
  "by_category": {
    "history": {
      "total": 5,
      "correct": 4,
      "invalid": 0,
      "failed": 0,
      "accuracy": 0.8
    },
    "computer science": {
      "total": 5,
      "correct": 3,
      "invalid": 0,
      "failed": 0,
      "accuracy": 0.6
    },
    "business": {
      "total": 5,
      "correct": 4,
      "invalid": 0,
      "failed": 0,
      "accuracy": 0.8
    },
    "biology": {
      "total": 5,
      "correct": 4,
      "invalid": 0,
      "failed": 0,
      "accuracy": 0.8
    },
    "psychology": {
      "total": 5,
      "correct": 5,
      "invalid": 0,
      "failed": 0,
      "accuracy": 1.0
    },
    "chemistry": {
      "total": 5,
      "correct": 2,
      "invalid": 0,
      "failed": 0,
      "accuracy": 0.4
    },
    "economics": {
      "total": 5,
      "correct": 3,
      "invalid": 0,
      "failed": 0,
      "accuracy": 0.6
    },
    "engineering": {
      "total": 5,
      "correct": 1,
      "invalid": 0,
      "failed": 0,
      "accuracy": 0.2
    },
    "other": {
      "total": 5,
      "correct": 5,
      "invalid": 0,
      "failed": 0,
      "accuracy": 1.0
    },
    "physics": {
      "total": 5,
      "correct": 3,
      "invalid": 0,
      "failed": 0,
      "accuracy": 0.6
    },
    "law": {
      "total": 5,
      "correct": 3,
      "invalid": 0,
      "failed": 0,
      "accuracy": 0.6
    },
    "health": {
      "total": 5,
      "correct": 4,
      "invalid": 0,
      "failed": 0,
      "accuracy": 0.8
    },
    "math": {
      "total": 5,
      "correct": 3,
      "invalid": 0,
      "failed": 0,
      "accuracy": 0.6
    },
    "philosophy": {
      "total": 5,
      "correct": 4,
      "invalid": 0,
      "failed": 0,
      "accuracy": 0.8
    }
  },
  "macro_accuracy": 0.6857142857142857
}
```

## 指标与限制

TTFT/TPOT 是客户端 SSE 观测值。输入吞吐以整个请求批次墙钟为分母，不是隔离的纯 Prefill 算子吞吐。
HBM 为5秒采样的观测峰值，不等价于瞬时 allocator 峰值。独立工作区未能从日志拆分时不编造数值。
细分结果见 cells.csv；图覆盖与显存日志见 server-evidence.json；Profiler 与15分钟稳定性原始数据分别在 profiles/ 和 stability/。