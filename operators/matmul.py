"""
Matrix Multiplication Operator - 矩阵乘法算子
基于Triton-Ascend实现的昇腾亲和矩阵乘法算子

优化策略：
1. 分块计算：减少内存访问次数
2. 向量化计算：使用tl.dot进行矩阵乘法
3. 多核并行：充分利用昇腾NPU多核
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    矩阵乘法核心内核
    
    计算公式: C = A @ B
    其中 A: (M, K), B: (K, N), C: (M, N)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        
        b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        
        acc += tl.dot(a, b)
    
    c = acc.to(tl.float16)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             c, mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    矩阵乘法算子入口函数
    
    Args:
        a: 输入矩阵A，形状为(M, K)
        b: 输入矩阵B，形状为(K, N)
    
    Returns:
        c: 输出矩阵C，形状为(M, N)
    """
    assert a.shape[1] == b.shape[0], f"Shape mismatch for matmul: a.shape={a.shape}, b.shape={b.shape}"
    
    a = a.contiguous()
    b = b.contiguous()
    
    M, K = a.shape
    K2, N = b.shape
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    if not has_npu_driver():
        return torch.matmul(a, b)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    
    if M >= 256 and N >= 256 and K >= 256:
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
    
    if M >= 512 and N >= 512 and K >= 512:
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return c


def matmul_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现"""
    return torch.matmul(a, b)


if __name__ == "__main__":
    print("Matrix Multiplication Operator Test")
    print("=" * 50)
    
    configs = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]
    
    for M, K, N in configs:
        a = torch.randn((M, K), device='npu:0' if has_npu_driver() else 'cpu', dtype=torch.float16)
        b = torch.randn((K, N), device=a.device, dtype=torch.float16)
        
        output = matmul(a, b)
        expected = torch.matmul(a.cpu(), b.cpu()).to(a.device)
        
        max_diff = torch.max(torch.abs(output.cpu() - expected.cpu()))
        print(f"Shape: ({M:4d},{K:4d})x({K:4d},{N:4d}) | Max diff: {max_diff:.6f}")
    
    print("Test completed successfully")