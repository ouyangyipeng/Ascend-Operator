"""
Vector Addition Operator - 向量加法算子
基于Triton-Ascend实现的昇腾亲和向量加法算子

优化策略：
1. 多核并行：充分利用昇腾NPU 32个AI Core（跨步分配）
2. 大块大小：减少内核启动开销，提高带宽利用率
3. 向量化内存访问：使用最大向量化加载
4. 数据对齐：确保BLOCK_SIZE是32的倍数
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


# Ascend 910B4 每个芯片有32个AI Core
NUM_AI_CORES = 32


@triton.jit
def _vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    向量加法核心内核 - 多核并行优化版本
    
    使用跨步分配策略，让每个AI Core处理不同的数据块，
    充分利用昇腾NPU的多核并行能力。
    """
    pid = tl.program_id(axis=0)
    NUM_CORES = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n_elements, BLOCK_SIZE)
    
    # 跨步分配：每个核处理 pid, pid+NUM_CORES, pid+2*NUM_CORES, ... 的块
    # 这样可以充分利用多核并行，同时保持良好的缓存局部性
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # 向量化加载和存储
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _vector_add_kernel_large(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    向量加法内核 - 大数据量优化版本
    
    使用更大的BLOCK_SIZE和循环展开，减少内核启动开销
    """
    pid = tl.program_id(axis=0)
    NUM_CORES = tl.num_programs(0)
    
    # 每个核处理的总元素数
    elements_per_core = tl.cdiv(n_elements, NUM_CORES)
    core_start = pid * elements_per_core
    
    # 处理该核负责的所有数据
    for offset_base in range(0, elements_per_core, BLOCK_SIZE):
        offsets = core_start + offset_base + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    向量加法算子入口函数
    
    Args:
        x: 输入张量
        y: 输入张量，形状与x相同
        
    Returns:
        output: x + y的结果
    """
    output = torch.empty_like(x)
    assert x.shape == y.shape, f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}"
    
    n_elements = output.numel()
    
    # CPU回退模式
    if not has_npu_driver():
        return x + y
    
    # 根据数据量选择最优配置
    # 注意：BLOCK_SIZE不能太大，否则会导致UB溢出
    if n_elements < 16384:
        # 小数据量：使用较少的核和较小的块
        BLOCK_SIZE = 1024
        grid = (min(8, triton.cdiv(n_elements, BLOCK_SIZE)),)
    elif n_elements < 262144:  # 256K
        # 中等数据量
        BLOCK_SIZE = 2048
        grid = (min(16, triton.cdiv(n_elements, BLOCK_SIZE)),)
    elif n_elements < 1048576:  # 1M
        # 较大数据量
        BLOCK_SIZE = 4096
        grid = (min(NUM_AI_CORES, triton.cdiv(n_elements, BLOCK_SIZE)),)
    else:
        # 大数据量：使用全部32个AI Core
        BLOCK_SIZE = 4096  # 使用适中的BLOCK_SIZE避免UB溢出
        grid = (NUM_AI_CORES,)
    
    # 启动内核
    _vector_add_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def vector_add_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现"""
    return x + y


if __name__ == "__main__":
    print("Vector Addition Operator Test")
    print("=" * 50)
    
    sizes = [1024, 10000, 100000, 1000000, 10000000]
    
    for size in sizes:
        x = torch.randn(size, device='npu:0' if has_npu_driver() else 'cpu')
        y = torch.randn(size, device=x.device)
        
        output = vector_add(x, y)
        expected = x + y
        
        max_diff = torch.max(torch.abs(output - expected))
        print(f"Size: {size:10d} | Max diff: {max_diff:.6f}")
    
    print("Test completed successfully")