"""
Vector Addition Operator - 向量加法算子
基于Triton-Ascend实现的昇腾亲和向量加法算子

优化策略：
1. Auto-tuning：自动搜索最优BLOCK_SIZE
2. 多核并行：充分利用昇腾NPU多核
3. 向量化内存访问：提高带宽利用率
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_stages=4, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def _vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    向量加法核心内核 - Auto-tuned版本
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    向量加法算子入口函数
    
    Args:
        x: 输入向量X，形状为(N,)
        y: 输入向量Y，形状为(N,)
    
    Returns:
        output: 输出向量，形状为(N,)
    """
    output = torch.empty_like(x)
    assert x.shape == y.shape, f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}"
    
    n_elements = output.numel()
    
    # 检查是否有NPU驱动
    if not has_npu_driver():
        # CPU模式：回退到PyTorch（仅用于开发调试）
        return x + y
    
    # NPU模式：使用Triton内核
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    _vector_add_kernel[grid](
        x, y, output,
        n_elements,
    )
    
    return output


def vector_add_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    PyTorch参考实现
    """
    return x + y


if __name__ == "__main__":
    print("Vector Addition Operator Test")
    print("=" * 50)
    
    # 测试不同大小
    sizes = [1024, 10000, 100000, 1000000, 10000000]
    
    for size in sizes:
        x = torch.randn(size, device='npu:0' if has_npu_driver() else 'cpu')
        y = torch.randn(size, device=x.device)
        
        output = vector_add(x, y)
        expected = x + y
        
        max_diff = torch.max(torch.abs(output - expected))
        print(f"Size: {size:10d} | Max diff: {max_diff:.6f}")
    
    print("Test completed successfully")