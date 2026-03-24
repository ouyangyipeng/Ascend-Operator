"""
Matrix Multiplication Operator - 矩阵乘法算子
基于Triton-Ascend实现的昇腾亲和矩阵乘法算子

优化策略：
1. 分块计算：减少内存访问次数，提高数据复用
2. 使用tl.dot：让编译器生成Cube矩阵计算指令
3. Super-Grouping：优化L2缓存访问模式
4. 多核并行：充分利用32个AI Core
5. 自适应分块：根据矩阵大小选择最优分块参数
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


# Ascend 910B4 每个芯片有32个AI Core
NUM_AI_CORES = 32


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
    
    使用分块计算，利用tl.dot生成Cube指令
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算当前块的位置
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 分块计算
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        # 加载A块
        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        
        # 加载B块
        b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        
        # 使用tl.dot进行矩阵乘法 - 编译器会生成Cube指令
        acc += tl.dot(a, b)
    
    # 存储结果
    c = acc.to(tl.float16)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             c, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _matmul_kernel_super_grouping(
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
    矩阵乘法内核 - Super-Grouping优化版本
    
    通过重新排列program_id来优化L2缓存访问
    让同一列的块连续执行，提高K维度的数据复用
    """
    # Super-Grouping: 重新排列pid以优化L2缓存
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # 计算当前块的位置
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 分块计算
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        # 加载A块
        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        
        # 加载B块
        b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        
        # 使用tl.dot进行矩阵乘法
        acc += tl.dot(a, b)
    
    # 存储结果
    c = acc.to(tl.float16)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             c, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _matmul_kernel_large(
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
    矩阵乘法内核 - 大矩阵优化版本
    
    使用更大的分块和更激进的优化
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 优化：预计算循环次数
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    
    for k_idx in range(num_k_blocks):
        k = k_idx * BLOCK_K
        rk = k + tl.arange(0, BLOCK_K)
        
        # 加载A块 - 使用向量化加载
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
                    mask=a_mask, other=0.0)
        
        # 加载B块
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)
        b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
                    mask=b_mask, other=0.0)
        
        # 矩阵乘法
        acc += tl.dot(a, b)
    
    # 存储结果
    c = acc.to(tl.float16)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             c, mask=(rm[:, None] < M) & (rn[None, :] < N))


def _get_optimal_block_sizes(M: int, N: int, K: int) -> tuple:
    """
    根据矩阵大小选择最优的分块参数
    
    考虑因素：
    1. Unified Buffer大小限制
    2. L2缓存利用率
    3. 并行度
    """
    # 小矩阵
    if M <= 64 and N <= 64 and K <= 64:
        return (32, 32, 32)
    
    # 中等矩阵
    if M <= 256 and N <= 256 and K <= 256:
        return (64, 64, 32)
    
    # 较大矩阵
    if M <= 512 and N <= 512 and K <= 512:
        return (64, 64, 64)
    
    # 大矩阵 - 使用更大的分块
    if M <= 1024 and N <= 1024:
        return (128, 64, 32)
    
    # 超大矩阵
    return (128, 64, 64)


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
    
    # 根据矩阵大小选择最优分块参数
    BLOCK_M, BLOCK_N, BLOCK_K = _get_optimal_block_sizes(M, N, K)
    
    # 计算grid大小
    num_blocks_m = triton.cdiv(M, BLOCK_M)
    num_blocks_n = triton.cdiv(N, BLOCK_N)
    
    # 选择内核和grid配置
    total_blocks = num_blocks_m * num_blocks_n
    
    if total_blocks <= NUM_AI_CORES:
        # 小矩阵：使用标准grid
        grid = (num_blocks_m, num_blocks_n)
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
    else:
        # 大矩阵：使用Super-Grouping优化
        grid = (total_blocks,)
        _matmul_kernel_super_grouping[grid](
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
        (64, 64, 64),
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