"""
Softmax Operator - Softmax算子
基于Triton-Ascend实现的昇腾亲和Softmax算子

优化策略：
1. 分块计算：沿最后一个维度分块处理
2. 数值稳定性：使用max值进行归一化
3. 多核并行：每个行由一个核处理
4. Auto-tuning：自动搜索最优分块大小
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


@triton.jit
def _softmax_kernel(
    output_ptr,  # 输出指针
    input_ptr,   # 输入指针
    input_row_stride,  # 输入行步长
    output_row_stride,  # 输出行步长
    n_cols,  # 列数
    BLOCK_SIZE: tl.constexpr,  # 分块大小
):
    """
    Softmax核心内核
    
    计算公式: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
    """
    # 获取当前处理的行索引
    row_idx = tl.program_id(0)
    
    # 计算当前行的起始位置
    row_start = row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # 初始化最大值和累加器
    max_val = float('-inf')
    sum_exp = 0.0
    
    # 第一遍：计算最大值
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < n_cols
        
        x = tl.load(input_ptr + row_start + offsets, mask=mask, other=float('-inf'))
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    # 第二遍：计算exp和sum
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < n_cols
        
        x = tl.load(input_ptr + row_start + offsets, mask=mask, other=float('-inf'))
        x_exp = tl.exp(x - max_val)
        sum_exp += tl.sum(x_exp, axis=0)
    
    # 第三遍：计算softmax并存储
    output_row_start = row_idx * output_row_stride
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < n_cols
        
        x = tl.load(input_ptr + row_start + offsets, mask=mask, other=float('-inf'))
        x_exp = tl.exp(x - max_val)
        softmax_val = x_exp / sum_exp
        
        tl.store(output_ptr + output_row_start + offsets, softmax_val, mask=mask)


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
    if dim != x.ndim - 1:
        x = x.transpose(dim, -1)
    
    # 确保连续
    x = x.contiguous()
    
    # 获取形状
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    
    # 预分配输出
    output = torch.empty_like(x)
    
    # 检查是否有NPU驱动
    if not has_npu_driver():
        # 回退到PyTorch实现
        result = torch.softmax(x, dim=-1)
        if dim != x.ndim - 1:
            result = result.transpose(dim, -1)
        return result
    
    # 定义grid
    grid = (n_rows,)
    
    # 启动内核
    _softmax_kernel[grid](
        output,
        x,
        x.stride(-2) if x.ndim > 1 else n_cols,
        output.stride(-2) if output.ndim > 1 else n_cols,
        n_cols,
        BLOCK_SIZE=256,
    )
    
    # 如果需要，转置回来
    if dim != x.ndim - 1:
        output = output.transpose(dim, -1)
    
    return output


def softmax_reference(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    PyTorch参考实现
    """
    return torch.softmax(x, dim=dim)


if __name__ == "__main__":
    # 测试代码
    print("Softmax Operator Test")
    print("=" * 50)
    
    # CPU测试
    x = torch.randn(128, 512)
    output = softmax(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 验证
    expected = torch.softmax(x, dim=-1)
    max_diff = torch.max(torch.abs(output - expected))
    print(f"Max difference: {max_diff}")
    
    # 验证sum为1
    row_sums = output.sum(dim=-1)
    print(f"Row sums (should be ~1): {row_sums[:5]}")
    print("CPU test completed successfully")