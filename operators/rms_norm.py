"""
RMS Normalization Operator - RMS Normalization算子
基于Triton-Ascend实现的昇腾亲和RMS Normalization算子

优化策略：
1. 融合内核：单块处理整行，融合归一化和权重应用
2. 多核并行：每行由一个核处理，充分利用32个AI Core
3. 向量化计算：利用Vector单元加速
4. 单遍计算：一次遍历完成平方和计算
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


# Ascend 910B4 每个芯片有32个AI Core
NUM_AI_CORES = 32


@triton.jit
def _rms_norm_kernel_fused(
    output_ptr,
    input_ptr,
    weight_ptr,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMS Normalization融合内核 - 单块处理整行
    
    计算公式: 
        rms = sqrt(mean(x^2) + eps)
        output = x / rms * weight
    
    适用于n_cols <= BLOCK_SIZE的情况
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 加载整行
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    
    # 计算平方和 - 单遍计算
    sum_sq = tl.sum(x * x, axis=0)
    mean_sq = sum_sq / n_cols
    
    # 计算RMS
    rms = tl.sqrt(mean_sq + eps)
    
    # 归一化并应用权重 - 融合操作
    output = (x / rms) * w
    
    tl.store(output_ptr + row_start + cols, output, mask=mask)


@triton.jit
def _rms_norm_kernel_blocked(
    output_ptr,
    input_ptr,
    weight_ptr,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMS Normalization内核 - 分块处理大列数
    
    第一遍：计算平方和
    第二遍：归一化并存储
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 第一遍：计算平方和
    sum_sq = 0.0
    count = 0.0
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
        count += tl.sum(mask.to(tl.float32), axis=0)
    
    # 计算RMS
    mean_sq = sum_sq / count
    rms = tl.sqrt(mean_sq + eps)
    
    # 第二遍：归一化并存储
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        
        output = (x / rms) * w
        tl.store(output_ptr + row_start + cols, output, mask=mask)


@triton.jit
def _rms_norm_kernel_vectorized(
    output_ptr,
    input_ptr,
    weight_ptr,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMS Normalization内核 - 向量化优化版本
    
    针对小列数进行向量化优化
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    # 向量化加载
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    
    # 向量化计算
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    
    # 融合归一化和权重应用
    output = x / rms * w
    
    tl.store(output_ptr + row_start + cols, output, mask=mask)


def _get_optimal_block_size(n_cols: int) -> int:
    """根据列数选择最优的BLOCK_SIZE"""
    if n_cols <= 64:
        return 64
    elif n_cols <= 128:
        return 128
    elif n_cols <= 256:
        return 256
    elif n_cols <= 512:
        return 512
    elif n_cols <= 1024:
        return 1024
    elif n_cols <= 2048:
        return 2048
    else:
        return 4096


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    RMS Normalization算子入口函数
    
    Args:
        x: 输入张量，形状为(..., N)
        weight: 缩放参数，形状为(N,)
        eps: 数值稳定性参数
    
    Returns:
        output: 归一化后的张量
    """
    x = x.contiguous()
    N = x.shape[-1]
    M = x.numel() // N
    output = torch.empty_like(x)
    
    # CPU模式：仅用于开发调试
    if not has_npu_driver():
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return (x / rms) * weight
    
    # 选择最优块大小
    BLOCK_SIZE = _get_optimal_block_size(N)
    
    # 根据数据量选择grid配置
    if M <= NUM_AI_CORES:
        grid = (M,)
    else:
        grid = (min(M, NUM_AI_CORES * 4),)
    
    if N <= BLOCK_SIZE:
        # 使用融合内核
        _rms_norm_kernel_fused[grid](
            output, x, weight,
            N,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # 使用分块内核
        _rms_norm_kernel_blocked[grid](
            output, x, weight,
            N,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output


def rms_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """PyTorch参考实现"""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


if __name__ == "__main__":
    print("RMS Normalization Operator Test")
    print("=" * 50)
    
    # 测试不同大小
    configs = [
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ]
    
    for rows, cols in configs:
        x = torch.randn((rows, cols), device='npu:0' if has_npu_driver() else 'cpu')
        weight = torch.ones(cols, device=x.device)
        
        output = rms_norm(x, weight)
        expected = rms_norm_reference(x.cpu(), weight.cpu()).to(x.device)
        
        max_diff = torch.max(torch.abs(output.cpu() - expected.cpu()))
        print(f"Shape: ({rows:4d},{cols:4d}) | Max diff: {max_diff:.6f}")
    
    print("Test completed successfully")