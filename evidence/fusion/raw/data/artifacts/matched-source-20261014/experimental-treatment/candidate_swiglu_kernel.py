from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_swiglu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BLOCKS_N: tl.constexpr,
):
    block_index = tl.program_id(0)
    m_block = block_index // NUM_BLOCKS_N
    n_block = block_index % NUM_BLOCKS_N
    m_indices = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_indices = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_indices < M
    n_mask = n_indices < N

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_offset in range(0, K, BLOCK_K):
        k_indices = k_offset + tl.arange(0, BLOCK_K)
        k_mask = k_indices < K
        a_vals = tl.load(
            a_ptr + m_indices[:, None] * stride_am + k_indices[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        gate_ptrs = (
            b_ptr + n_indices[:, None] * stride_bn + k_indices[None, :] * stride_bk
        )
        gate_vals = tl.load(
            gate_ptrs,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        up_vals = tl.load(
            gate_ptrs + N * stride_bn,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        gate_acc += tl.dot(a_vals, tl.trans(gate_vals))
        up_acc += tl.dot(a_vals, tl.trans(up_vals))

    # Preserve the materialized GEMM's FP16/BF16 rounding boundary.
    # Fusion must not silently change the activation's input precision.
    gate = gate_acc.to(a_ptr.dtype.element_ty).to(tl.float32)
    up = up_acc.to(a_ptr.dtype.element_ty).to(tl.float32)
    output = gate * tl.sigmoid(gate) * up
    tl.store(
        c_ptr + m_indices[:, None] * stride_cm + n_indices[None, :] * stride_cn,
        output,
        mask=m_mask[:, None] & n_mask[None, :],
    )


def linear_swiglu(x: torch.Tensor, gate_up_weight: torch.Tensor) -> torch.Tensor:
    """Fused gate/up linear and SwiGLU for validated M <= 128 shapes."""
    if x.ndim != 2 or gate_up_weight.ndim != 2:
        raise ValueError("linear_swiglu expects two 2D tensors")
    if x.shape[0] > 128:
        raise ValueError("linear_swiglu fused kernel is limited to M <= 128")
    if x.shape[1] != gate_up_weight.shape[1] or gate_up_weight.shape[0] % 2:
        raise ValueError("incompatible gate/up linear shapes")
    if x.dtype not in (torch.float16, torch.bfloat16) or x.dtype != gate_up_weight.dtype:
        raise TypeError("linear_swiglu supports matching FP16/BF16 inputs")

    m, k = x.shape
    n = gate_up_weight.shape[0] // 2
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0 or n == 0:
        return output
    block_m = min(triton.next_power_of_2(m), 128)
    block_n = 128
    num_blocks_m = triton.cdiv(m, block_m)
    num_blocks_n = triton.cdiv(n, block_n)
    _linear_swiglu_kernel[(num_blocks_m * num_blocks_n,)](
        x,
        gate_up_weight,
        output,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        gate_up_weight.stride(0),
        gate_up_weight.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=256,
        NUM_BLOCKS_N=num_blocks_n,
    )
    return output
