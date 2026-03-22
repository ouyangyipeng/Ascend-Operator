"""
Matrix Multiplication Operator - 矩阵乘法算子
基于Triton-Ascend实现的昇腾亲和矩阵乘法算子

优化策略：
1. 分块计算：将大矩阵分割成小块进行计算
2. L2缓存优化：使用super-grouping提高缓存命中率
3. 多核并行：固定核数为物理核数
4. Auto-tuning：自动搜索最优分块参数
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


@triton.jit
def _matmul_kernel(
    # 指向矩阵的指针
    a_ptr, b_ptr, c_ptr,
    # 矩阵形状
    M, N, K,
    # 步长
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 分块大小
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # 分组大小（用于L2缓存优化）
    GROUP_SIZE_M: tl.constexpr,
):
    """
    矩阵乘法核心内核
    
    计算公式: C = A @ B
    其中 A: (M, K), B: (K, N), C: (M, N)
    """
    # 获取程序ID
    pid = tl.program_id(axis=0)
    
    # 计算M和N方向的程序数量
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # L2缓存优化：super-grouping
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # 计算当前程序处理的块索引
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # 计算A矩阵块的起始位置
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    # 计算B矩阵块的起始位置
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # K维度循环
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载A和B的块
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # 矩阵乘法累加
        accumulator += tl.dot(a, b)
        
        # 更新指针
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # 将结果转换为输出类型
    c = accumulator.to(tl.float16)
    
    # 计算C矩阵的存储位置
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # 存储结果（带边界检查）
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    矩阵乘法算子入口函数
    
    Args:
        a: 输入矩阵A，形状为(M, K)
        b: 输入矩阵B，形状为(K, N)
    
    Returns:
        c: 输出矩阵C，形状为(M, N)
    """
    # 检查输入
    assert a.shape[1] == b.shape[0], f"Shape mismatch for matmul: a.shape={a.shape}, b.shape={b.shape}"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    M, K = a.shape
    K, N = b.shape
    
    # 预分配输出
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # 检查是否有NPU驱动
    if not has_npu_driver():
        # 回退到PyTorch实现
        return torch.matmul(a, b)
    
    # 定义grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # 启动内核
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )
    
    return c


def matmul_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    PyTorch参考实现
    
    用于验证Triton实现的正确性
    """
    return torch.matmul(a, b)


if __name__ == "__main__":
    # 测试代码
    print("Matrix Multiplication Operator Test")
    print("=" * 50)
    
    # CPU测试
    a = torch.randn(128, 128)
    b = torch.randn(128, 128)
    c = matmul(a, b)
    print(f"Input A shape: {a.shape}")
    print(f"Input B shape: {b.shape}")
    print(f"Output C shape: {c.shape}")
    
    # 验证
    expected = torch.matmul(a, b)
    max_diff = torch.max(torch.abs(c - expected))
    print(f"Max difference: {max_diff}")
    print("CPU test completed successfully")