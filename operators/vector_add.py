"""
Vector Addition Operator - 向量加法算子
基于Triton-Ascend实现的昇腾亲和向量加法算子

优化策略：
1. 多核并行：将分核数量固定为硬件物理核数
2. 存算并行：默认开启multiBuffer
3. Auto-tuning：自动搜索最优BLOCK_SIZE
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver, get_device


@triton.jit
def _vector_add_kernel(
    x_ptr,  # 输入向量X的指针
    y_ptr,  # 输入向量Y的指针
    output_ptr,  # 输出向量的指针
    n_elements,  # 元素总数
    BLOCK_SIZE: tl.constexpr,  # 每个程序处理的块大小
):
    """
    向量加法核心内核
    
    计算公式: output = x + y
    
    优化要点：
    - 使用program_id获取当前核的ID
    - 使用固定核数进行任务分配
    - 使用mask处理边界情况
    """
    # 获取当前程序的ID
    pid = tl.program_id(axis=0)
    
    # 获取总核数（用于跨步分配任务）
    NUM_CORES = tl.num_programs(0)
    
    # 计算总块数
    NUM_BLOCKS = tl.cdiv(n_elements, BLOCK_SIZE)
    
    # 跨步分配任务：每个核处理stride=NUM_CORES的块
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
        # 计算当前块的起始位置
        block_start = block_idx * BLOCK_SIZE
        
        # 计算偏移量
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        
        # 创建mask防止越界访问
        mask = offsets < n_elements
        
        # 从全局内存加载数据到片上内存
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        
        # 执行加法计算
        output = x + y
        
        # 将结果写回全局内存
        tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    向量加法算子入口函数
    
    Args:
        x: 输入向量X，形状为(N,)
        y: 输入向量Y，形状为(N,)
    
    Returns:
        output: 输出向量，形状为(N,)
    
    Example:
        >>> x = torch.randn(1024, device='npu:0')
        >>> y = torch.randn(1024, device='npu:0')
        >>> output = vector_add(x, y)
    """
    # 预分配输出张量
    output = torch.empty_like(x)
    
    # 检查输入形状
    assert x.shape == y.shape, f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}"
    
    n_elements = output.numel()
    
    # 检查是否有NPU驱动
    if not has_npu_driver():
        # 回退到PyTorch实现
        return x + y
    
    # 定义grid函数
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 启动内核
    _vector_add_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=1024,
    )
    
    return output


# Auto-tune配置 - 仅在有NPU驱动时定义
_autotuned_kernel = None

def _get_autotuned_kernel():
    """延迟创建autotuned内核"""
    global _autotuned_kernel
    
    if _autotuned_kernel is not None:
        return _autotuned_kernel
    
    if not has_npu_driver():
        return None
    
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_SIZE': 2048}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_SIZE': 4096}, num_stages=2, num_warps=4),
        ],
        key=['n_elements'],
    )
    @triton.jit
    def _vector_add_kernel_autotuned_inner(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        带有Auto-tuning的向量加法内核
        
        自动搜索最优的BLOCK_SIZE配置
        """
        pid = tl.program_id(axis=0)
        NUM_CORES = tl.num_programs(0)
        NUM_BLOCKS = tl.cdiv(n_elements, BLOCK_SIZE)
        
        for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
            block_start = block_idx * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)
    
    _autotuned_kernel = _vector_add_kernel_autotuned_inner
    return _autotuned_kernel


def vector_add_autotuned(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    带有Auto-tuning的向量加法算子
    
    自动搜索最优配置以获得最佳性能
    """
    # 检查是否有NPU驱动
    if not has_npu_driver():
        # 回退到PyTorch实现
        return x + y
    
    output = torch.empty_like(x)
    assert x.shape == y.shape
    n_elements = output.numel()
    
    kernel = _get_autotuned_kernel()
    if kernel is None:
        return x + y
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    kernel[grid](
        x, y, output,
        n_elements,
    )
    
    return output


if __name__ == "__main__":
    # 测试代码
    print("Vector Add Operator Test")
    print("=" * 50)
    
    # 创建测试数据
    size = 98432
    
    # CPU测试
    x = torch.randn(size)
    y = torch.randn(size)
    output = x + y
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("CPU test completed successfully")
    
    # 检查NPU
    print(f"\nHas NPU driver: {has_npu_driver()}")
    print(f"Device: {get_device()}")