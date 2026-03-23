#!/usr/bin/env python3
"""
Triton-Ascend 算子测试
测试所有实现的算子在NPU上的运行
"""

import torch
import torch_npu
import triton
import triton.language as tl
import time

print("=" * 60)
print("Triton-Ascend 算子测试")
print("=" * 60)


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """向量加法内核"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def test_vector_add():
    """测试向量加法"""
    print("\n=== 向量加法测试 ===")
    n = 1024 * 1024  # 1M elements
    x = torch.randn(n, device="npu:0", dtype=torch.float32)
    y = torch.randn(n, device="npu:0", dtype=torch.float32)
    output = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    
    # Warmup
    vector_add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    torch.npu.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        vector_add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    torch.npu.synchronize()
    elapsed = time.time() - start
    
    # Verify
    expected = x + y
    max_diff = torch.max(torch.abs(output.cpu() - expected.cpu())).item()
    
    print(f"  Size: {n} elements ({n * 4 / 1024**2:.2f} MB)")
    print(f"  Time: {elapsed * 1000 / 100:.3f} ms per iteration")
    print(f"  Bandwidth: {n * 4 * 3 / elapsed / 1024**3:.2f} GB/s")
    print(f"  Max diff: {max_diff}")
    print(f"  Result: {'SUCCESS' if max_diff < 1e-5 else 'FAILED'}")
    return max_diff < 1e-5


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    """Softmax内核"""
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + row_start + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_ptrs = output_ptr + row_start + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def test_softmax():
    """测试Softmax"""
    print("\n=== Softmax测试 ===")
    rows, cols = 1024, 1024
    x = torch.randn((rows, cols), device="npu:0", dtype=torch.float32)
    output = torch.empty_like(x)
    
    grid = (rows,)
    
    # Warmup
    softmax_kernel[grid](output, x, x.stride(0), rows, cols, BLOCK_SIZE=1024)
    torch.npu.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        softmax_kernel[grid](output, x, x.stride(0), rows, cols, BLOCK_SIZE=1024)
    torch.npu.synchronize()
    elapsed = time.time() - start
    
    # Verify
    expected = torch.softmax(x, dim=1)
    max_diff = torch.max(torch.abs(output.cpu() - expected.cpu())).item()
    
    print(f"  Size: ({rows}, {cols})")
    print(f"  Time: {elapsed * 1000 / 100:.3f} ms per iteration")
    print(f"  Max diff: {max_diff}")
    print(f"  Result: {'SUCCESS' if max_diff < 1e-4 else 'FAILED'}")
    return max_diff < 1e-4


@triton.jit
def layernorm_kernel(output_ptr, input_ptr, weight_ptr, bias_ptr, 
                     n_rows, n_cols, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """LayerNorm内核"""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load row
    row_ptr = input_ptr + row_idx * n_cols
    row = tl.load(row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean and variance
    mean = tl.sum(row, axis=0) / n_cols
    diff = row - mean
    variance = tl.sum(diff * diff, axis=0) / n_cols
    
    # Normalize
    row_norm = (row - mean) / tl.sqrt(variance + eps)
    
    # Apply weight and bias
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    output = row_norm * weight + bias
    
    # Store
    output_ptr = output_ptr + row_idx * n_cols
    tl.store(output_ptr + col_offsets, output, mask=mask)


def test_layernorm():
    """测试LayerNorm"""
    print("\n=== LayerNorm测试 ===")
    rows, cols = 1024, 1024
    x = torch.randn((rows, cols), device="npu:0", dtype=torch.float32)
    weight = torch.randn(cols, device="npu:0", dtype=torch.float32)
    bias = torch.randn(cols, device="npu:0", dtype=torch.float32)
    output = torch.empty_like(x)
    
    grid = (rows,)
    eps = 1e-5
    
    # Warmup
    layernorm_kernel[grid](output, x, weight, bias, rows, cols, eps=eps, BLOCK_SIZE=1024)
    torch.npu.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        layernorm_kernel[grid](output, x, weight, bias, rows, cols, eps=eps, BLOCK_SIZE=1024)
    torch.npu.synchronize()
    elapsed = time.time() - start
    
    # Verify
    expected = torch.nn.functional.layer_norm(x, (cols,), weight, bias, eps=eps)
    max_diff = torch.max(torch.abs(output.cpu() - expected.cpu())).item()
    
    print(f"  Size: ({rows}, {cols})")
    print(f"  Time: {elapsed * 1000 / 100:.3f} ms per iteration")
    print(f"  Max diff: {max_diff}")
    print(f"  Result: {'SUCCESS' if max_diff < 1e-4 else 'FAILED'}")
    return max_diff < 1e-4


if __name__ == "__main__":
    results = {}
    
    try:
        results["vector_add"] = test_vector_add()
    except Exception as e:
        print(f"向量加法测试失败: {e}")
        results["vector_add"] = False
    
    try:
        results["softmax"] = test_softmax()
    except Exception as e:
        print(f"Softmax测试失败: {e}")
        results["softmax"] = False
    
    try:
        results["layernorm"] = test_layernorm()
    except Exception as e:
        print(f"LayerNorm测试失败: {e}")
        results["layernorm"] = False
    
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {name}: {status}")
    print("=" * 60)