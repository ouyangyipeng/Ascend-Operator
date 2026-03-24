"""
Flash Attention Operator - Flash Attention算子
基于Triton-Ascend实现的昇腾亲和Flash Attention算子

优化策略：
1. 分块计算：将Q、K、V分块处理，减少内存访问
2. 在线Softmax：避免存储大型注意力矩阵
3. 多核并行：沿batch和head维度并行
4. 使用tl.dot：让编译器生成Cube矩阵计算指令
5. 优化分块大小：避免UB/CBUF溢出
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


# Ascend 910B4 每个芯片有32个AI Core
NUM_AI_CORES = 32


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
    Flash Attention核心内核 - 优化版本
    
    使用在线Softmax算法，避免存储大型注意力矩阵
    使用tl.dot进行矩阵乘法，让编译器生成Cube指令
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
    q_ptrs = q_ptr + q_offset + q_rows[:, None] * stride_qs + tl.arange(0, BLOCK_D)[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    
    # 遍历K、V块
    for k_start in range(0, seq_len, BLOCK_N):
        k_rows = k_start + tl.arange(0, BLOCK_N)
        k_mask = k_rows < seq_len
        
        # 加载K块
        k_offset = pid_b * stride_kb + pid_h * stride_kh
        k_ptrs = k_ptr + k_offset + k_rows[:, None] * stride_ks + tl.arange(0, BLOCK_D)[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)
        
        # 计算QK^T - 使用tl.dot让编译器生成Cube指令
        qk = tl.dot(q, tl.trans(k)) * scale
        
        # 处理无效位置（设置为-inf）
        qk = tl.where(q_mask[:, None] & k_mask[None, :], qk, float('-inf'))
        
        # 在线Softmax更新
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        
        # 防止数值问题：当m_new为-inf时，exp会出问题
        # 使用safe_exp：当m_new为-inf时，alpha为0
        alpha = tl.where(m_i > float('-inf'), tl.exp(m_i - m_new), 0.0)
        p = tl.where(q_mask[:, None] & k_mask[None, :], tl.exp(qk - m_new[:, None]), 0.0)
        
        # 加载V块
        v_offset = pid_b * stride_vb + pid_h * stride_vh
        v_ptrs = v_ptr + v_offset + k_rows[:, None] * stride_vs + tl.arange(0, BLOCK_D)[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)
        
        # 更新累加器 - 使用tl.dot
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
    
    # 归一化并存储
    # 防止除以0
    l_i = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_i[:, None]
    
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    o_ptrs = o_ptr + o_offset + q_rows[:, None] * stride_os + tl.arange(0, BLOCK_D)[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=q_mask[:, None])


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    Flash Attention算子入口函数
    
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
        # 使用标准注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)
    
    # NPU模式：根据输入大小选择最优配置
    # BLOCK_D需要能整除head_dim
    if head_dim <= 32:
        BLOCK_D = 32
    elif head_dim <= 64:
        BLOCK_D = 64
    elif head_dim <= 128:
        BLOCK_D = 128
    else:
        BLOCK_D = 64  # 默认使用较小的块
    
    # 根据seq_len选择BLOCK_M和BLOCK_N
    if seq_len <= 64:
        BLOCK_M = min(64, seq_len)
        BLOCK_N = min(64, seq_len)
    elif seq_len <= 128:
        BLOCK_M = min(64, seq_len)
        BLOCK_N = min(64, seq_len)
    elif seq_len <= 256:
        BLOCK_M = 64
        BLOCK_N = 32
    elif seq_len <= 512:
        BLOCK_M = 64
        BLOCK_N = 64
    else:
        BLOCK_M = 32
        BLOCK_N = 32
    
    # 标准内核
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
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


if __name__ == "__main__":
    print("Flash Attention Operator Test")
    print("=" * 50)
    
    # 测试不同大小
    configs = [
        (1, 8, 64, 64),
        (1, 8, 128, 64),
        (2, 4, 256, 32),
        (1, 8, 512, 64),
    ]
    
    for batch, heads, seq_len, head_dim in configs:
        q = torch.randn((batch, heads, seq_len, head_dim), 
                       device='npu:0' if has_npu_driver() else 'cpu', 
                       dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        output = flash_attention(q, k, v)
        expected = flash_attention_reference(q.cpu(), k.cpu(), v.cpu()).to(q.device)
        
        max_diff = torch.max(torch.abs(output.cpu() - expected.cpu()))
        print(f"Shape: ({batch},{heads},{seq_len},{head_dim}) | Max diff: {max_diff:.6f}")
    
    print("Test completed successfully")