"""
Vector Addition Operator - 向量加法算子
基于Triton-Ascend实现的昇腾亲和向量加法算子

优化策略：
1. 多核并行：充分利用昇腾NPU多核（跨步分配）
2. 大块大小：减少内核启动次数
3. 向量化内存访问：提高带宽利用率
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


def get_npu_num_cores():
    """获取NPU的AI Core数量"""
    try:
        import torch_npu
        if torch.npu.is_available():
            return 32
    except:
        pass
    return 1


@triton.jit
def _vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """向量加法核心内核 - 多核并行版本"""
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


@triton.jit
def _vector_add_simple_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """向量加法简单内核 - 用于小数据量"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """向量加法算子入口函数"""
    output = torch.empty_like(x)
    assert x.shape == y.shape, f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}"
    
    n_elements = output.numel()
    
    if not has_npu_driver():
        return x + y
    
    NUM_CORES = get_npu_num_cores()
    
    if n_elements < 8192:
        BLOCK_SIZE = max(1024, triton.cdiv(n_elements, NUM_CORES))
        grid = (min(NUM_CORES, triton.cdiv(n_elements, BLOCK_SIZE)),)
        _vector_add_simple_kernel[grid](
            x, y, output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        BLOCK_SIZE = 4096
        NUM_BLOCKS = triton.cdiv(n_elements, BLOCK_SIZE)
        grid = (min(NUM_CORES, NUM_BLOCKS),)
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