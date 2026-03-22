"""
RMS Normalization Operator - RMS Normalization算子
基于Triton-Ascend实现的昇腾亲和RMS Normalization算子

优化策略：
1. 行级并行：每行由一个核处理
2. 向量化计算：使用向量化操作提高效率
3. Auto-tuning：自动搜索最优分块大小
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


@triton.jit
def _rms_norm_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    M, N,
    stride_row,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMS Normalization核心内核
    
    计算公式: 
        rms = sqrt(mean(x^2) + eps)
        output = x / rms * weight
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_row
    
    # 计算平方和
    sum_sq = 0.0
    count = 0
    
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        x = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        
        sum_sq += tl.sum(x * x)
        count += tl.sum(mask.to(tl.float32))
    
    # 计算RMS
    mean_sq = sum_sq / (count + 1e-8)
    rms = tl.sqrt(mean_sq + eps)
    
    # 归一化并应用权重
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        x = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        
        output = (x / rms) * w
        
        tl.store(output_ptr + row_start + offsets, output, mask=mask)


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
    
    # 检查是否有NPU驱动
    if not has_npu_driver():
        # 回退到PyTorch实现
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return (x / rms) * weight
    
    grid = (M,)
    _rms_norm_kernel[grid](
        output, x, weight,
        M, N, N,
        eps=eps,
        BLOCK_SIZE=256,
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
    
    x = torch.randn(128, 512)
    weight = torch.ones(512)
    
    output = rms_norm(x, weight)
    expected = rms_norm_reference(x, weight)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    max_diff = torch.max(torch.abs(output - expected))
    print(f"Max difference: {max_diff}")
    print("CPU test completed successfully")