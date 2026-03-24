"""
Reduction Operators - 归约算子
基于Triton-Ascend实现的昇腾亲和归约算子

优化策略：
1. 多核并行：跨行并行处理，充分利用32个AI Core
2. 向量化归约：利用Vector单元进行高效的归约操作
3. 分块处理：处理大矩阵时使用分块策略
4. 融合内核：减少内核启动开销
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


# Ascend 910B4 每个芯片有32个AI Core
NUM_AI_CORES = 32


@triton.jit
def _reduce_sum_rows_kernel(
    output_ptr,
    input_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    行求和内核 - 每行由一个核处理
    
    计算公式: output[i] = sum(input[i, :])
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    row_start = row_idx * n_cols
    sum_val = 0.0
    
    # 分块处理每行
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
    
    tl.store(output_ptr + row_idx, sum_val)


@triton.jit
def _reduce_sum_rows_fused_kernel(
    output_ptr,
    input_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    行求和融合内核 - 单块处理整行
    
    适用于n_cols <= BLOCK_SIZE的情况
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    row_start = row_idx * n_cols
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    sum_val = tl.sum(x, axis=0)
    
    tl.store(output_ptr + row_idx, sum_val)


@triton.jit
def _reduce_max_rows_kernel(
    output_ptr,
    input_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    行最大值内核 - 每行由一个核处理
    
    计算公式: output[i] = max(input[i, :])
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    row_start = row_idx * n_cols
    max_val = float('-inf')
    
    # 分块处理每行
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('-inf')).to(tl.float32)
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    tl.store(output_ptr + row_idx, max_val)


@triton.jit
def _reduce_max_rows_fused_kernel(
    output_ptr,
    input_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    行最大值融合内核 - 单块处理整行
    
    适用于n_cols <= BLOCK_SIZE的情况
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    row_start = row_idx * n_cols
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('-inf')).to(tl.float32)
    max_val = tl.max(x, axis=0)
    
    tl.store(output_ptr + row_idx, max_val)


@triton.jit
def _reduce_min_rows_kernel(
    output_ptr,
    input_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    行最小值内核 - 每行由一个核处理
    
    计算公式: output[i] = min(input[i, :])
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    row_start = row_idx * n_cols
    min_val = float('inf')
    
    # 分块处理每行
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('inf')).to(tl.float32)
        block_min = tl.min(x, axis=0)
        min_val = tl.minimum(min_val, block_min)
    
    tl.store(output_ptr + row_idx, min_val)


@triton.jit
def _reduce_min_rows_fused_kernel(
    output_ptr,
    input_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    行最小值融合内核 - 单块处理整行
    
    适用于n_cols <= BLOCK_SIZE的情况
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    row_start = row_idx * n_cols
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('inf')).to(tl.float32)
    min_val = tl.min(x, axis=0)
    
    tl.store(output_ptr + row_idx, min_val)


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


def reduce_sum(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    求和归约算子
    
    Args:
        x: 输入张量
        dim: 归约维度，默认为最后一个维度
        keepdim: 是否保持维度
    
    Returns:
        output: 归约结果
    """
    if dim < 0:
        dim = x.ndim + dim
    
    # 确保在最后一个维度上归约
    need_transpose = dim != x.ndim - 1
    if need_transpose:
        x = x.transpose(dim, -1)
    
    x = x.contiguous()
    
    # 获取形状
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    
    # CPU模式
    if not has_npu_driver():
        result = torch.sum(x, dim=-1, keepdim=keepdim)
        if need_transpose:
            result = result.transpose(dim, -1)
        return result
    
    # 预分配输出
    if keepdim:
        output = torch.empty((n_rows, 1), device=x.device, dtype=x.dtype)
    else:
        output = torch.empty((n_rows,), device=x.device, dtype=x.dtype)
    
    # 选择最优块大小
    BLOCK_SIZE = _get_optimal_block_size(n_cols)
    
    # 根据数据量选择grid配置
    if n_rows <= NUM_AI_CORES:
        grid = (n_rows,)
    else:
        grid = (min(n_rows, NUM_AI_CORES * 4),)
    
    if n_cols <= BLOCK_SIZE:
        # 使用融合内核
        _reduce_sum_rows_fused_kernel[grid](
            output, x, n_rows, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # 使用分块内核
        _reduce_sum_rows_kernel[grid](
            output, x, n_rows, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # 处理维度
    if keepdim:
        output = output.reshape(n_rows, 1)
    
    if need_transpose:
        output = output.transpose(dim, -1)
    
    return output


def reduce_max(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    最大值归约算子
    
    Args:
        x: 输入张量
        dim: 归约维度，默认为最后一个维度
        keepdim: 是否保持维度
    
    Returns:
        output: 归约结果
    """
    if dim < 0:
        dim = x.ndim + dim
    
    need_transpose = dim != x.ndim - 1
    if need_transpose:
        x = x.transpose(dim, -1)
    
    x = x.contiguous()
    
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    
    if not has_npu_driver():
        result = torch.max(x, dim=-1, keepdim=keepdim)[0]
        if need_transpose:
            result = result.transpose(dim, -1)
        return result
    
    if keepdim:
        output = torch.empty((n_rows, 1), device=x.device, dtype=x.dtype)
    else:
        output = torch.empty((n_rows,), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = _get_optimal_block_size(n_cols)
    
    if n_rows <= NUM_AI_CORES:
        grid = (n_rows,)
    else:
        grid = (min(n_rows, NUM_AI_CORES * 4),)
    
    if n_cols <= BLOCK_SIZE:
        _reduce_max_rows_fused_kernel[grid](
            output, x, n_rows, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        _reduce_max_rows_kernel[grid](
            output, x, n_rows, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    if keepdim:
        output = output.reshape(n_rows, 1)
    
    if need_transpose:
        output = output.transpose(dim, -1)
    
    return output


def reduce_min(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    最小值归约算子
    
    Args:
        x: 输入张量
        dim: 归约维度，默认为最后一个维度
        keepdim: 是否保持维度
    
    Returns:
        output: 归约结果
    """
    if dim < 0:
        dim = x.ndim + dim
    
    need_transpose = dim != x.ndim - 1
    if need_transpose:
        x = x.transpose(dim, -1)
    
    x = x.contiguous()
    
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    
    if not has_npu_driver():
        result = torch.min(x, dim=-1, keepdim=keepdim)[0]
        if need_transpose:
            result = result.transpose(dim, -1)
        return result
    
    if keepdim:
        output = torch.empty((n_rows, 1), device=x.device, dtype=x.dtype)
    else:
        output = torch.empty((n_rows,), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = _get_optimal_block_size(n_cols)
    
    if n_rows <= NUM_AI_CORES:
        grid = (n_rows,)
    else:
        grid = (min(n_rows, NUM_AI_CORES * 4),)
    
    if n_cols <= BLOCK_SIZE:
        _reduce_min_rows_fused_kernel[grid](
            output, x, n_rows, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        _reduce_min_rows_kernel[grid](
            output, x, n_rows, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    if keepdim:
        output = output.reshape(n_rows, 1)
    
    if need_transpose:
        output = output.transpose(dim, -1)
    
    return output


if __name__ == "__main__":
    print("Reduction Operators Test")
    print("=" * 50)
    
    configs = [
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ]
    
    for rows, cols in configs:
        x = torch.randn((rows, cols), device='npu:0' if has_npu_driver() else 'cpu')
        
        # Test sum
        output_sum = reduce_sum(x)
        expected_sum = torch.sum(x, dim=-1)
        max_diff_sum = torch.max(torch.abs(output_sum.cpu() - expected_sum.cpu()))
        
        # Test max
        output_max = reduce_max(x)
        expected_max = torch.max(x, dim=-1)[0]
        max_diff_max = torch.max(torch.abs(output_max.cpu() - expected_max.cpu()))
        
        # Test min
        output_min = reduce_min(x)
        expected_min = torch.min(x, dim=-1)[0]
        max_diff_min = torch.max(torch.abs(output_min.cpu() - expected_min.cpu()))
        
        print(f"Shape: ({rows:4d},{cols:4d}) | Sum diff: {max_diff_sum:.6f} | Max diff: {max_diff_max:.6f} | Min diff: {max_diff_min:.6f}")
    
    print("Test completed successfully")