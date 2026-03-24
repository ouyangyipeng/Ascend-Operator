"""
Softmax Operator - Softmax算子
基于Triton-Ascend实现的昇腾亲和Softmax算子

优化策略：
1. 融合内核：单块处理整行
2. 数值稳定性：使用max值进行归一化
3. 多核并行：每行由一个核处理
4. 自适应配置：根据输入大小选择最优BLOCK_SIZE
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


@triton.jit
def _softmax_kernel(
    output_ptr,
    input_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softmax融合内核 - 单块处理整行
    适用于n_cols <= BLOCK_SIZE的情况
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 加载整行
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('-inf')).to(tl.float32)
    
    # 数值稳定性：减去最大值
    max_val = tl.max(x, axis=0)
    x_shifted = x - max_val
    
    # 计算exp和sum
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    
    # 归一化
    softmax_val = exp_x / sum_exp
    
    # 存储
    tl.store(output_ptr + row_start + cols, softmax_val, mask=mask)


@triton.jit
def _softmax_kernel_large(
    output_ptr,
    input_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softmax内核 - 用于大列数
    分块处理每行
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 第一遍：计算最大值
    max_val = float('-inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('-inf')).to(tl.float32)
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    # 第二遍：计算exp和sum
    sum_exp = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('-inf')).to(tl.float32)
        exp_x = tl.exp(x - max_val)
        sum_exp += tl.sum(exp_x, axis=0)
    
    # 第三遍：计算并存储结果
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('-inf')).to(tl.float32)
        exp_x = tl.exp(x - max_val)
        softmax_val = exp_x / sum_exp
        tl.store(output_ptr + row_start + cols, softmax_val, mask=mask)


def _get_optimal_block_size(n_cols: int) -> int:
    """
    根据列数选择最优的BLOCK_SIZE
    """
    if n_cols <= 128:
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


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax算子入口函数
    
    Args:
        x: 输入张量
        dim: 计算softmax的维度，默认为最后一个维度
    
    Returns:
        output: Softmax结果
    """
    # 处理负数维度
    if dim < 0:
        dim = x.ndim + dim
    
    # 确保在最后一个维度上计算
    need_transpose = dim != x.ndim - 1
    if need_transpose:
        x = x.transpose(dim, -1)
    
    # 确保连续
    x = x.contiguous()
    
    # 获取形状
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    
    # 预分配输出
    output = torch.empty_like(x)
    
    # CPU模式：仅用于开发调试
    if not has_npu_driver():
        result = torch.softmax(x, dim=-1)
        if need_transpose:
            result = result.transpose(dim, -1)
        return result
    
    # NPU模式：选择最优块大小
    BLOCK_SIZE = _get_optimal_block_size(n_cols)
    
    # 启动内核
    grid = (n_rows,)
    
    if n_cols <= BLOCK_SIZE:
        # 使用融合内核
        _softmax_kernel[grid](
            output, x, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # 使用分块内核
        _softmax_kernel_large[grid](
            output, x, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # 如果需要转置，转回来
    if need_transpose:
        output = output.transpose(dim, -1)
    
    return output


def softmax_reference(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """PyTorch参考实现"""
    return torch.softmax(x, dim=dim)


if __name__ == "__main__":
    print("Softmax Operator Test")
    print("=" * 50)
    
    # 测试不同大小
    configs = [
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ]
    
    for rows, cols in configs:
        x = torch.randn((rows, cols), device='npu:0' if has_npu_driver() else 'cpu')
        
        output = softmax(x)
        expected = torch.softmax(x, dim=-1)
        
        max_diff = torch.max(torch.abs(output.cpu() - expected.cpu()))
        print(f"Shape: ({rows:4d},{cols:4d}) | Max diff: {max_diff:.6f}")
    
    print("Test completed successfully")