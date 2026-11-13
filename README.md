# vLLM-HUST Weight Placement Research and Experimental Results

Author: **Zhou Shenghui (周升晖)**.

Model: **Qwen3.5-35B-A3B**. Hardware: **two Ascend 910B2 NPUs**. This repository contains experimental results from October 2026.

This project investigates weight placement and parallel configurations for dual-NPU inference with vLLM-HUST. It records the configuration, raw results, and analysis of the B0 TP2 baseline and W1 TP2+EP2 screening experiments. Related shared-expert operator fusion and Graph task-group switching experiments are also included, with their validation scope and conclusions documented separately.

## Experiment Dates

All dates below use Beijing time (Asia/Shanghai, UTC+8).

| Experiment | Date |
|---|---|
| Shared-expert fusion adaptation, numerical fixes, and three-round performance validation | October 14, 2026 |
| Graph task-group switching probes | October 15, 2026 |
| B0 weight-placement baseline | October 23–24, 2026; completed at 00:47 on October 24 |
| W1 TP2+EP2 Small screening | October 26, 2026 |

Timestamps ending in `Z` in the original logs are UTC; add eight hours to convert them to Beijing time.

## Reading Guide

1. [Experimental results and data index](docs/RESULTS_INDEX.md): conclusions, original files, and scope.
2. [Contributions](docs/CONTRIBUTIONS.md): work performed by Zhou Shenghui and the corresponding records.
3. [Data completeness and verification methodology](docs/METHODOLOGY.md): data availability and verification steps.
4. [Offline verification results](verification/results.json): key metrics recomputed from request-level records or collected summaries.
5. [Provenance and SHA-256 manifest](provenance.json): original workspace paths, file sizes, and hashes.

## Results Overview

| Experiment | Observations | Conclusion |
|---|---|---|
| Shared-expert MatMul–SwiGLU fusion | Three independent baseline/fusion service pairs; mean C1 TPOT improved by 1.97% for short prompts and 0.67% for long prompts; 13/20 quality questions answered correctly in every round | Adaptation and numerical fixes validated; a stable end-to-end speedup has not been established |
| zeros→empty | Counterexamples observed with production shapes and NaN/nonzero output prefilling | Disabled in the tested environment; this does not establish failure in every environment |
| Graph task-group switching | Three compatible single-task probes passed; task-chain replacement failed | Validated switching for restricted structures; arbitrary subgraph hot switching remains unimplemented |
| B0 weight-placement baseline | 90 Medium test cells: 87 completed and 3 warmup timeouts; three-round mean P2 C64 throughput of 944.92 output tokens/s; quality score of 1414/2406 | Fixed-configuration TP2 baseline completed; optimal weight placement has not been established |
| W1 TP2+EP2 | All 12 Small performance cells and 384 requests completed; geometric mean throughput ratio changed by −1.80% relative to historical B0 | No clear benefit in single-round screening across different test periods; Mini/Medium testing was not started |

The fusion experiments labeled B0/B1 and the weight-placement experiments labeled B0/W1 are separate experiments. Small and Medium use different scheduling budgets. Large batch throughput is excluded from Medium statistics.

## Files and Verification

`evidence/` preserves the original files. `tools/` provides offline verification scripts, and `docs/` explains the results and methodology. Absolute paths, remote paths, and relative links in historical documents may refer to the original workspace; use this repository's data index to locate archived files. Historical plans do not establish that an experiment was executed: in particular, the W1 Mini/Medium scripts were preparation only.

```bash
python tools/verify_evidence.py
```

The complete fusion archive, `opset-repair-evidence-20261014.tar.gz` (approximately 121 MB), contains Profiler CSV files, databases, and traces. It is distributed as a [Release attachment](https://github.com/vLLM-HUST/vllm-hust-weight-layout-research/releases/tag/experiments-2026-10), rather than as a regular Git file. Until the upload completes, the archive is available locally under `release-assets/`; attachment availability is determined by the actual Release contents.

SHA-256: `91a8967a208da284d13ce4c749332f238f3e58e3679ffe64299889c283eaab8d`.
