"""
Reduction Operators - 归约算子集合
基于Triton-Ascend实现的昇腾亲和归约算子

包含：
- reduce_sum: 求和归约
- reduce_max: 最大值归约
- reduce_min: 最小值归约

优化策略：
1. 多核并行：充分利用昇腾NPU多核
2. 向量化计算：使用向量化操作
3. 树形归约：减少同步开销
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


@triton.jit
def _reduce_sum_rows_kernel(
    output_ptr,
    input_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    行求和内核 - 每行由一个核处理
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 分块求和
    sum_val = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
    
    tl.store(output_ptr + row_idx, sum_val)


@triton.jit
def _reduce_max_rows_kernel(
    output_ptr,
    input_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    行最大值内核 - 每行由一个核处理
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 分块求最大值
    max_val = float('-inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('-inf')).to(tl.float32)
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    tl.store(output_ptr + row_idx, max_val)


@triton.jit
def _reduce_min_rows_kernel(
    output_ptr,
    input_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    行最小值内核 - 每行由一个核处理
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 分块求最小值
    min_val = float('inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('inf')).to(tl.float32)
        block_min = tl.min(x, axis=0)
        min_val = tl.minimum(min_val, block_min)
    
    tl.store(output_ptr + row_idx, min_val)


def _get_optimal_block_size(n_cols: int) -> int:
    """根据列数选择最优的BLOCK_SIZE"""
    if n_cols <= 1024:
        return 1024
    elif n_cols <= 2048:
        return 2048
    else:
        return 4096


def reduce_sum(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    求和归约
    
    Args:
        x: 输入张量
        dim: 归约维度
        keepdim: 是否保持维度
    
    Returns:
        归约结果
    """
    if dim < 0:
        dim = x.ndim + dim
    
    # 简化处理：只支持最后一维
    if dim != x.ndim - 1:
        return torch.sum(x, dim=dim, keepdim=keepdim)
    
    # CPU模式
    if not has_npu_driver():
        return torch.sum(x, dim=dim, keepdim=keepdim)
    
    x = x.contiguous()
    n_cols = x.shape[-1]
    M = x.numel() // n_cols
    
    # 预分配输出
    output_shape = list(x.shape)
    if keepdim:
        output_shape[dim] = 1
    else:
        output_shape = output_shape[:-1]
    
    output = torch.zeros(output_shape, device=x.device, dtype=torch.float32)
    
    # 选择块大小
    BLOCK_SIZE = _get_optimal_block_size(n_cols)
    
    # 启动内核
    grid = (M,)
    _reduce_sum_rows_kernel[grid](
        output, x, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # 转换回原始dtype
    output = output.to(x.dtype)
    
    return output


def reduce_max(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    最大值归约
    
    Args:
        x: 输入张量
        dim: 归约维度
        keepdim: 是否保持维度
    
    Returns:
        归约结果
    """
    if dim < 0:
        dim = x.ndim + dim
    
    # 简化处理：只支持最后一维
    if dim != x.ndim - 1:
        return torch.max(x, dim=dim, keepdim=keepdim)[0]
    
    # CPU模式
    if not has_npu_driver():
        return torch.max(x, dim=dim, keepdim=keepdim)[0]
    
    x = x.contiguous()
    n_cols = x.shape[-1]
    M = x.numel() // n_cols
    
    # 预分配输出
    output_shape = list(x.shape)
    if keepdim:
        output_shape[dim] = 1
    else:
        output_shape = output_shape[:-1]
    
    output = torch.full(output_shape, float('-inf'), device=x.device, dtype=torch.float32)
    
    # 选择块大小
    BLOCK_SIZE = _get_optimal_block_size(n_cols)
    
    # 启动内核
    grid = (M,)
    _reduce_max_rows_kernel[grid](
        output, x, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # 转换回原始dtype
    output = output.to(x.dtype)
    
    return output


def reduce_min(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    最小值归约
    
    Args:
        x: 输入张量
        dim: 归约维度
        keepdim: 是否保持维度
    
    Returns:
        归约结果
    """
    if dim < 0:
        dim = x.ndim + dim
    
    # 简化处理：只支持最后一维
    if dim != x.ndim - 1:
        return torch.min(x, dim=dim, keepdim=keepdim)[0]
    
    # CPU模式
    if not has_npu_driver():
        return torch.min(x, dim=dim, keepdim=keepdim)[0]
    
    x = x.contiguous()
    n_cols = x.shape[-1]
    M = x.numel() // n_cols
    
    # 预分配输出
    output_shape = list(x.shape)
    if keepdim:
        output_shape[dim] = 1
    else:
        output_shape = output_shape[:-1]
    
    output = torch.full(output_shape, float('inf'), device=x.device, dtype=torch.float32)
    
    # 选择块大小
    BLOCK_SIZE = _get_optimal_block_size(n_cols)
    
    # 启动内核
    grid = (M,)
    _reduce_min_rows_kernel[grid](
        output, x, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # 转换回原始dtype
    output = output.to(x.dtype)
    
    return output


if __name__ == "__main__":
    print("Reduction Operators Test")
    print("=" * 50)
    
    device = 'npu:0' if has_npu_driver() else 'cpu'
    x = torch.randn(128, 512, device=device)
    
    print(f"Input shape: {x.shape}")
    
    # 测试sum
    output_sum = reduce_sum(x, dim=-1)
    expected_sum = torch.sum(x, dim=-1)
    max_diff_sum = torch.max(torch.abs(output_sum - expected_sum))
    print(f"Sum max diff: {max_diff_sum:.6f}")
    
    # 测试max
    output_max = reduce_max(x, dim=-1)
    expected_max = torch.max(x, dim=-1)[0]
    max_diff_max = torch.max(torch.abs(output_max - expected_max))
    print(f"Max max diff: {max_diff_max:.6f}")
    
    # 测试min
    output_min = reduce_min(x, dim=-1)
    expected_min = torch.min(x, dim=-1)[0]
    max_diff_min = torch.max(torch.abs(output_min - expected_min))
    print(f"Min max diff: {max_diff_min:.6f}")
    
    print("Test completed successfully")