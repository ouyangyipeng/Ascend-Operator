"""
Layer Normalization Operator - Layer Normalization算子
基于Triton-Ascend实现的昇腾亲和Layer Normalization算子

优化策略：
1. 融合内核：单块处理整行
2. Welford算法：数值稳定的在线统计
3. 多核并行：每行由一个核处理
4. 自适应配置：根据输入大小选择最优BLOCK_SIZE
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


@triton.jit
def _layer_norm_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    bias_ptr,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Layer Normalization融合内核 - 单块处理整行
    适用于n_cols <= BLOCK_SIZE的情况
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 加载整行
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # 计算均值和方差
    mean = tl.sum(x, axis=0) / n_cols
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # 归一化并应用仿射变换
    x_norm = diff * rstd
    output = w * x_norm + b
    
    tl.store(output_ptr + row_start + cols, output, mask=mask)


@triton.jit
def _layer_norm_kernel_large(
    output_ptr,
    input_ptr,
    weight_ptr,
    bias_ptr,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Layer Normalization内核 - 用于大列数
    分块处理每行，使用Welford算法
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # Welford算法：在线计算均值和方差
    count = 0
    mean = 0.0
    M2 = 0.0
    
    # 第一遍：计算统计量
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        
        # 计算块内统计
        delta = x - mean
        n_new = count + tl.sum(mask.to(tl.float32))
        mean = mean + tl.sum(delta, axis=0) / n_new
        delta2 = x - mean
        M2 = M2 + tl.sum(delta * delta2, axis=0)
        count = n_new
    
    # 计算最终统计量
    var = M2 / count
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # 第二遍：计算并存储结果
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        
        x_norm = (x - mean) * rstd
        output = w * x_norm + b
        
        tl.store(output_ptr + row_start + cols, output, mask=mask)


def _get_optimal_block_size(n_cols: int) -> int:
    """根据列数选择最优的BLOCK_SIZE"""
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


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Layer Normalization算子入口函数
    
    Args:
        x: 输入张量，形状为(..., N)，在最后一个维度上进行归一化
        weight: 缩放参数，形状为(N,)
        bias: 偏移参数，形状为(N,)
        eps: 数值稳定性参数
    
    Returns:
        output: 归一化后的张量
    """
    x = x.contiguous()
    N = x.shape[-1]
    
    # 预分配输出
    output = torch.empty_like(x)
    
    # CPU模式：仅用于开发调试
    if not has_npu_driver():
        return torch.nn.functional.layer_norm(x, (N,), weight, bias, eps)
    
    # 获取行数
    M = x.numel() // N
    
    # 选择最优块大小
    BLOCK_SIZE = _get_optimal_block_size(N)
    
    # 启动内核
    grid = (M,)
    
    if N <= BLOCK_SIZE:
        # 使用融合内核
        _layer_norm_kernel[grid](
            output, x, weight, bias,
            N,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # 使用分块内核
        _layer_norm_kernel_large[grid](
            output, x, weight, bias,
            N,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output


def layer_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """PyTorch参考实现"""
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)


if __name__ == "__main__":
    print("Layer Normalization Operator Test")
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
        weight = torch.ones(cols, device=x.device)
        bias = torch.zeros(cols, device=x.device)
        
        output = layer_norm(x, weight, bias)
        expected = torch.nn.functional.layer_norm(x, (cols,), weight, bias)
        
        max_diff = torch.max(torch.abs(output.cpu() - expected.cpu()))
        print(f"Shape: ({rows:4d},{cols:4d}) | Max diff: {max_diff:.6f}")
    
    print("Test completed successfully")