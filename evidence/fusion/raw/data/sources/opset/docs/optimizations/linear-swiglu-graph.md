# Linear + SwiGLU Graph fusion

## Mechanism

The baseline materializes the `[M, 2N]` gate/up result from an unquantized
GEMM and then launches `npu_swiglu`. The candidate keeps gate and up
accumulators in one persistent Triton kernel, applies
`gate * sigmoid(gate) * up`, and stores only `[M, N]`.

The validated path accepts `M <= 128`. Larger dynamic prefill shapes and
unknown graph structures fail closed to native GEMM plus `npu_swiglu`.

## Operator result

Qwen2.5-7B MLP projection (`K=3584`, `N=18944`, BF16), ten warmups and fifty
synchronized samples per arm:

| M | Baseline mean +/- SD | Fused mean +/- SD | Latency reduction |
| ---: | ---: | ---: | ---: |
| 1 | 0.5001 +/- 0.0115 ms | 0.4204 +/- 0.0083 ms | 15.93% |
| 32 | 0.4989 +/- 0.0126 ms | 0.4095 +/- 0.0100 ms | 17.92% |
| 128 | 0.5416 +/- 0.0111 ms | 0.4507 +/- 0.0068 ms | 16.79% |

The full 256-column fused tile exceeded CBUF capacity. A 128-column output
tile compiled successfully, showing that simultaneous FP32 accumulator
residency is the principal fusion constraint.

## End-to-end result

Qwen2.5-7B-Instruct, BF16, Ascend 910B2, CANN 9.0.0, Graph mode,
`VLLM_BATCH_INVARIANT=1`, ShareGPT 64 requests, concurrency 4 and output length
128. Three matched rounds followed an eight-request workload warmup.

| Metric | Baseline mean +/- SD | Candidate mean +/- SD | Improvement |
| --- | ---: | ---: | ---: |
| TPOT | 19.3481 +/- 0.5271 ms | 18.6088 +/- 0.1224 ms | 3.82% lower |
| Output token throughput | 187.5615 +/- 7.4216 token/s | 197.3891 +/- 1.4549 token/s | 5.24% higher |
| Total token throughput | 562.4271 +/- 22.2546 token/s | 591.4048 +/- 4.3592 token/s | 5.15% higher |
| Request throughput | 1.4979 +/- 0.0593 req/s | 1.5744 +/- 0.0116 req/s | 5.11% higher |
| TTFT | 219.4263 +/- 38.5764 ms | 180.4964 +/- 3.2113 ms | 17.74% lower |

All six formal runs completed 64/64 requests with no failures, and matched
correctness prompts produced identical text. See the compact
[`e2e-summary.json`](../../results/linear_swiglu_graph/e2e-summary.json).

## Boundary and negative evidence

- These results cover unquantized Graph/Triton execution, not ACLNN or
  quantized linear paths.
- A no-workload-warmup experiment exposed one-time Triton JIT cost and yielded
  negative candidate TPOT; cold start is deliberately separated from the
  steady-state claim.
- Large-M fusion caused CBUF overflow and, for some multi-tile shapes,
  incorrect values. The implementation therefore falls back for `M > 128`.
# Qwen3.5 MoE development validation (2026-10-14)

The shared expert is hidden inside `moe_forward_shared`, so an outer FX matcher
can legitimately report zero matches. A guarded internal shared-expert seam now
fuses its gate/up projection and SwiGLU while retaining caller stream/event
boundaries, down projection and expert gating. Kernel traces, rather than the
outer matcher count, establish whether this seam was exercised.

The candidate preserves FP16/BF16 GEMM rounding before activation. The internal
seam requires batch-invariant NPU execution, M=1..128, an unquantized bias-free
projection and supported activation; LoRA, enabled weight prefetch and larger M
fall back to the original path. Unknown split-method structures are rejected.

On Qwen3.5-35B-A3B TP2 Graph, three independent matched service pairs preserved
the 20-question MMLU-Pro sample accuracy (65%) and Transformer smoke output.
Changing-input operator Graph checks passed 42 comparisons. Single-request mean
TPOT reductions were 1.97% (short) and 0.67% (long); lifecycle variation is large
enough that these do not establish a stable general end-to-end gain. The initial
C2 measurements lacked C2-specific warmup and are exploratory only. This is not
full MMLU-Pro or five-dataset certification, and no package release is implied.

