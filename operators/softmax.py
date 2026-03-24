"""
Softmax Operator - Softmax算子
基于Triton-Ascend实现的昇腾亲和Softmax算子

优化策略：
1. Online Softmax：单遍计算max和sum，减少内存访问
2. 融合内核：单块处理整行，减少内核启动开销
3. 多核并行：每行由一个核处理，充分利用32个AI Core
4. 向量化计算：利用Vector单元加速exp计算
5. 数值稳定性：使用max值进行归一化防止溢出
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


# Ascend 910B4 每个芯片有32个AI Core
NUM_AI_CORES = 32


@triton.jit
def _softmax_kernel_fused(
    output_ptr,
    input_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softmax融合内核 - 单块处理整行
    
    适用于n_cols <= BLOCK_SIZE的情况
    使用单遍计算，减少内存访问
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 加载整行 - 使用实际的n_cols
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    # 加载数据，无效位置用-inf填充（不影响max计算）
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('-inf')).to(tl.float32)
    
    # 数值稳定性：减去最大值
    max_val = tl.max(x, axis=0)
    
    # 计算exp(x - max)
    x_shifted = x - max_val
    exp_x = tl.exp(x_shifted)
    
    # 计算sum
    sum_exp = tl.sum(exp_x, axis=0)
    
    # 归一化
    softmax_val = exp_x / sum_exp
    
    # 存储（只存储有效位置）
    tl.store(output_ptr + row_start + cols, softmax_val, mask=mask)


@triton.jit
def _softmax_kernel_online(
    output_ptr,
    input_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Online Softmax内核 - 用于大列数
    
    使用Online Softmax算法，两遍完成计算：
    第一遍：计算max和sum
    第二遍：计算并存储结果
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 第一遍：Online计算max和sum
    max_val = float('-inf')
    sum_exp = 0.0
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('-inf')).to(tl.float32)
        
        # Online update
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_val, block_max)
        
        # 更新sum_exp
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
        max_val = new_max
    
    # 第二遍：计算并存储结果
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
    确保BLOCK_SIZE >= n_cols时使用融合内核
    """
    # 选择能容纳整行的最小BLOCK_SIZE
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
    elif n_cols <= 4096:
        return 4096
    else:
        return 4096  # 使用分块处理


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
    
    # NPU模式：选择最优块大小和内核
    BLOCK_SIZE = _get_optimal_block_size(n_cols)
    
    # Grid配置：每个program处理一行，所以grid大小等于n_rows
    grid = (n_rows,)
    
    # 判断是否使用融合内核
    # 当BLOCK_SIZE >= n_cols时，可以使用融合内核
    if n_cols <= BLOCK_SIZE:
        # 使用融合内核（单遍）
        _softmax_kernel_fused[grid](
            output, x, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # 使用Online Softmax内核（两遍）
        _softmax_kernel_online[grid](
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
        (2048, 2048),
    ]
    
    for rows, cols in configs:
        x = torch.randn((rows, cols), device='npu:0' if has_npu_driver() else 'cpu')
        
        output = softmax(x)
        expected = torch.softmax(x, dim=-1)
        
        max_diff = torch.max(torch.abs(output.cpu() - expected.cpu()))
        print(f"Shape: ({rows:4d},{cols:4d}) | Max diff: {max_diff:.6f}")
    
    print("Test completed successfully")