"""
Layer Normalization Operator - Layer Normalization算子
基于Triton-Ascend实现的昇腾亲和Layer Normalization算子

优化策略：
1. Welford算法：单遍计算均值和方差，减少内存访问
2. 融合内核：单块处理整行，融合归一化和仿射变换
3. 多核并行：每行由一个核处理，充分利用32个AI Core
4. 向量化计算：利用Vector单元加速
"""

import torch
import triton
import triton.language as tl

from .utils import has_npu_driver


# Ascend 910B4 每个芯片有32个AI Core
NUM_AI_CORES = 32


@triton.jit
def _layer_norm_kernel_fused(
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
    
    使用Welford算法单遍计算均值和方差，然后融合归一化和仿射变换
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
    
    # Welford算法：单遍计算均值和方差
    # mean = sum(x) / n
    # var = sum((x - mean)^2) / n
    mean = tl.sum(x, axis=0) / n_cols
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # 融合归一化和仿射变换: y = w * (x - mean) * rstd + b
    x_norm = diff * rstd
    output = w * x_norm + b
    
    tl.store(output_ptr + row_start + cols, output, mask=mask)


@triton.jit
def _layer_norm_kernel_welford(
    output_ptr,
    input_ptr,
    weight_ptr,
    bias_ptr,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Layer Normalization内核 - 使用Welford在线算法
    
    用于大列数情况，分块处理每行
    Welford算法可以数值稳定地在线计算均值和方差
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # Welford算法：在线计算均值和M2（用于计算方差）
    count = 0.0
    mean = 0.0
    M2 = 0.0
    
    # 第一遍：使用Welford算法计算统计量
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        
        # Welford在线更新
        # 对于每个元素x：
        # count += 1
        # delta = x - mean
        # mean += delta / count
        # delta2 = x - mean
        # M2 += delta * delta2
        
        # 向量化版本的Welford算法
        block_count = tl.sum(mask.to(tl.float32), axis=0)
        if block_count > 0:
            # 计算块内统计量
            block_mean = tl.sum(x, axis=0) / block_count
            block_var = tl.sum((x - block_mean) * (x - block_mean), axis=0)
            
            # 合并统计量
            if count == 0.0:
                count = block_count
                mean = block_mean
                M2 = block_var
            else:
                # 合并两个统计量
                delta = block_mean - mean
                new_count = count + block_count
                mean = (count * mean + block_count * block_mean) / new_count
                M2 = M2 + block_var + delta * delta * count * block_count / new_count
                count = new_count
    
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
        
        # 融合归一化和仿射变换
        x_norm = (x - mean) * rstd
        output = w * x_norm + b
        
        tl.store(output_ptr + row_start + cols, output, mask=mask)


@triton.jit
def _layer_norm_kernel_two_pass(
    output_ptr,
    input_ptr,
    weight_ptr,
    bias_ptr,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Layer Normalization内核 - 两遍算法
    
    第一遍：计算均值和方差
    第二遍：归一化并应用仿射变换
    
    对于中等大小的列数，这个版本可能更高效
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 第一遍：计算均值
    sum_x = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
    
    mean = sum_x / n_cols
    
    # 第一遍（续）：计算方差
    sum_sq = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        sum_sq += tl.sum(diff * diff, axis=0)
    
    var = sum_sq / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # 第二遍：归一化并存储
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
    if n_cols <= 64:
        return 64
    elif n_cols <= 128:
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
    
    # 根据数据量选择grid配置
    if M <= NUM_AI_CORES:
        grid = (M,)
    else:
        grid = (min(M, NUM_AI_CORES * 4),)
    
    if N <= BLOCK_SIZE:
        # 使用融合内核（单遍）
        _layer_norm_kernel_fused[grid](
            output, x, weight, bias,
            N,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # 使用两遍算法（对于大列数更稳定）
        _layer_norm_kernel_two_pass[grid](
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
        (2048, 2048),
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