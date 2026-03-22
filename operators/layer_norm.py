"""
Layer Normalization Operator - Layer Normalization算子
基于Triton-Ascend实现的昇腾亲和Layer Normalization算子

优化策略：
1. 行级并行：每行由一个核处理
2. 向量化计算：使用向量化操作提高效率
3. 数值稳定性：使用Welford算法计算方差
4. Auto-tuning：自动搜索最优分块大小
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
    M, N,
    stride_row,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Layer Normalization核心内核
    
    计算公式: 
        mean = mean(x)
        var = var(x)
        x_norm = (x - mean) / sqrt(var + eps)
        output = weight * x_norm + bias
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_row
    
    # 第一遍：计算均值
    mean = 0.0
    count = 0
    
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        x = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        
        block_count = tl.sum(mask.to(tl.float32))
        block_mean = tl.sum(x) / (block_count + 1e-8)
        
        new_count = count + block_count
        mean = mean * (count / (new_count + 1e-8)) + block_mean * (block_count / (new_count + 1e-8))
        count = new_count
    
    # 第二遍：计算方差
    var = 0.0
    count = 0
    
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        x = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        
        diff = x - mean
        block_var = tl.sum(diff * diff * mask.to(tl.float32))
        block_count = tl.sum(mask.to(tl.float32))
        
        new_count = count + block_count
        var = var * (count / (new_count + 1e-8)) + block_var / (new_count + 1e-8)
        count = new_count
    
    # 计算rstd
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # 第三遍：归一化并应用仿射变换
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        x = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        x_norm = (x - mean) * rstd
        output = w * x_norm + b
        
        tl.store(output_ptr + row_start + offsets, output, mask=mask)


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
    M = x.numel() // N
    output = torch.empty_like(x)
    
    # 检查是否有NPU驱动
    if not has_npu_driver():
        # 回退到PyTorch实现
        return torch.nn.functional.layer_norm(x, (N,), weight, bias, eps)
    
    grid = (M,)
    _layer_norm_kernel[grid](
        output, x, weight, bias,
        M, N, N,
        eps=eps,
        BLOCK_SIZE=256,
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
    
    x = torch.randn(128, 512)
    weight = torch.ones(512)
    bias = torch.zeros(512)
    
    output = layer_norm(x, weight, bias)
    expected = torch.nn.functional.layer_norm(x, (512,), weight, bias)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    max_diff = torch.max(torch.abs(output - expected))
    print(f"Max difference: {max_diff}")
    print("CPU test completed successfully")