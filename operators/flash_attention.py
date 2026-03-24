"""
Flash Attention Operator - Flash Attention算子
基于Triton-Ascend实现的昇腾亲和Flash Attention算子

优化策略：
1. 分块计算：将Q、K、V分块处理，减少内存访问
2. 在线Softmax：避免存储大型注意力矩阵
3. 多核并行：沿序列长度维度并行
4. 数值稳定性：处理边界条件和NaN问题
5. 内存优化：使用较小的块大小避免缓冲区溢出
6. 自适应配置：根据输入大小选择最优块配置
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


@triton.jit
def _flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    seq_len, head_dim, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Flash Attention核心内核 - 内存优化版
    
    使用较小的块大小避免UB/CBUF溢出
    """
    pid_b = tl.program_id(0)  # batch
    pid_h = tl.program_id(1)  # head
    pid_m = tl.program_id(2)  # seq block
    
    # Q块的起始行
    q_start = pid_m * BLOCK_M
    q_rows = q_start + tl.arange(0, BLOCK_M)
    q_mask = q_rows < seq_len
    
    # 初始化累加器
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    
    # 加载Q块
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    q = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for d_off in range(0, head_dim, BLOCK_D):
        d_idx = d_off + tl.arange(0, BLOCK_D)
        d_mask = d_idx < head_dim
        q_ptrs = q_ptr + q_offset + q_rows[:, None] * stride_qs + d_idx[None, :] * stride_qd
        q_block = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
        if d_off == 0:
            q = q_block.to(tl.float32)
    
    # 遍历K、V块
    for k_start in range(0, seq_len, BLOCK_N):
        k_rows = k_start + tl.arange(0, BLOCK_N)
        k_mask = k_rows < seq_len
        
        # 加载K块
        k_offset = pid_b * stride_kb + pid_h * stride_kh
        k = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        for d_off in range(0, head_dim, BLOCK_D):
            d_idx = d_off + tl.arange(0, BLOCK_D)
            d_mask = d_idx < head_dim
            k_ptrs = k_ptr + k_offset + k_rows[:, None] * stride_ks + d_idx[None, :] * stride_kd
            k_block = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
            if d_off == 0:
                k = k_block.to(tl.float32)
        
        # 计算QK^T
        qk = tl.dot(q, tl.trans(k)) * scale
        
        # 在线Softmax
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])
        l_new = l_i * tl.exp(m_i - m_new) + tl.sum(p, axis=1)
        
        # 加载V块
        v_offset = pid_b * stride_vb + pid_h * stride_vh
        v = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        for d_off in range(0, head_dim, BLOCK_D):
            d_idx = d_off + tl.arange(0, BLOCK_D)
            d_mask = d_idx < head_dim
            v_ptrs = v_ptr + v_offset + k_rows[:, None] * stride_vs + d_idx[None, :] * stride_vd
            v_block = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
            if d_off == 0:
                v = v_block.to(tl.float32)
        
        # 更新累加器
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        
        m_i = m_new
        l_i = l_new
    
    # 归一化并存储
    acc = acc / l_i[:, None]
    
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    for d_off in range(0, head_dim, BLOCK_D):
        d_idx = d_off + tl.arange(0, BLOCK_D)
        d_mask = d_idx < head_dim
        o_ptrs = o_ptr + o_offset + q_rows[:, None] * stride_os + d_idx[None, :] * stride_od
        tl.store(o_ptrs, acc.to(tl.float16), mask=q_mask[:, None] & d_mask[None, :])


def _get_optimal_config(seq_len: int, head_dim: int) -> tuple:
    """
    根据输入大小选择最优配置
    
    Returns:
        (BLOCK_M, BLOCK_N, BLOCK_D, num_stages, num_warps)
    """
    # 小序列长度 - 使用小块大小增加并行度
    if seq_len <= 128:
        if head_dim <= 32:
            return (32, 32, 32, 2, 4)
        else:
            return (32, 32, 64, 2, 4)
    
    # 中等序列长度
    elif seq_len <= 512:
        if head_dim <= 32:
            return (64, 64, 32, 3, 8)
        else:
            return (32, 64, 64, 3, 4)
    
    # 大序列长度 - 使用更大的块大小
    else:
        if head_dim <= 32:
            return (128, 64, 32, 4, 8)
        else:
            return (64, 128, 64, 4, 8)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    Flash Attention算子入口函数 - 纯Triton实现
    
    根据输入大小自适应选择最优配置，无PyTorch回退
    
    Args:
        q: Query张量，形状为(batch, num_heads, seq_len, head_dim)
        k: Key张量，形状为(batch, num_heads, seq_len, head_dim)
        v: Value张量，形状为(batch, num_heads, seq_len, head_dim)
        scale: 缩放因子，默认为1/sqrt(head_dim)
    
    Returns:
        output: 注意力输出，形状为(batch, num_heads, seq_len, head_dim)
    """
    assert q.shape == k.shape == v.shape, f"Shape mismatch: q={q.shape}, k={k.shape}, v={v.shape}"
    
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    
    output = torch.empty_like(q)
    
    # CPU模式：仅用于开发调试
    if not has_npu_driver():
        # 使用默认配置
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_D = min(64, head_dim)
    else:
        # NPU模式：根据输入大小选择最优配置
        BLOCK_M, BLOCK_N, BLOCK_D, _, _ = _get_optimal_config(seq_len, head_dim)
    
    grid = (
        batch_size,
        num_heads,
        triton.cdiv(seq_len, BLOCK_M),
    )
    
    _flash_attn_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        seq_len, head_dim, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    
    return output


def flash_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """PyTorch参考实现"""
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    
    return output


if __name__ == "__main__":
    print("Flash Attention Operator Test")
    print("=" * 50)
    
    # 测试不同配置
    configs = [
        (1, 8, 128, 64),
        (2, 4, 256, 32),
        (1, 8, 512, 64),
    ]
    
    for batch, heads, seq_len, head_dim in configs:
        print(f"\nTesting config: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}")
        
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device='cpu')
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device='cpu')
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device='cpu')
        
        output = flash_attention(q, k, v)
        expected = flash_attention_reference(q, k, v)
        
        max_diff = torch.max(torch.abs(output - expected))
        print(f"Max difference: {max_diff}")
        
        if torch.isnan(output).any():
            print("ERROR: Output contains NaN!")
        else:
            print("OK: No NaN detected")