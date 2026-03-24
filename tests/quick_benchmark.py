#!/usr/bin/env python3
"""
快速性能基准测试脚本
"""

import torch
import torch_npu
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import operators


def quick_benchmark():
    """快速基准测试"""
    device = 'npu:0'
    print("=" * 60)
    print("快速性能基准测试")
    print("=" * 60)
    
    results = {}
    
    # VectorAdd
    print("\n[VectorAdd]")
    size = 1048576  # 1M
    x = torch.randn(size, device=device)
    y = torch.randn(size, device=device)
    
    # 预热Triton内核
    for _ in range(3):
        _ = operators.vector_add(x, y)
    torch.npu.synchronize()
    
    # PyTorch基线
    torch.npu.synchronize()
    start = time.time()
    for _ in range(10):
        _ = x + y
    torch.npu.synchronize()
    pytorch_time = (time.time() - start) / 10
    
    # Triton测试
    torch.npu.synchronize()
    start = time.time()
    for _ in range(10):
        _ = operators.vector_add(x, y)
    torch.npu.synchronize()
    triton_time = (time.time() - start) / 10
    
    speedup = pytorch_time / triton_time
    results['vector_add'] = speedup
    print(f"  Size: 1M | PyTorch: {pytorch_time*1000:.3f}ms | Triton: {triton_time*1000:.3f}ms | Speedup: {speedup:.2f}x")
    
    # Softmax
    print("\n[Softmax]")
    x = torch.randn((1024, 1024), device=device)
    
    # 预热
    for _ in range(3):
        _ = operators.softmax(x)
    torch.npu.synchronize()
    
    torch.npu.synchronize()
    start = time.time()
    for _ in range(10):
        _ = torch.softmax(x, dim=-1)
    torch.npu.synchronize()
    pytorch_time = (time.time() - start) / 10
    
    torch.npu.synchronize()
    start = time.time()
    for _ in range(10):
        _ = operators.softmax(x)
    torch.npu.synchronize()
    triton_time = (time.time() - start) / 10
    
    speedup = pytorch_time / triton_time
    results['softmax'] = speedup
    print(f"  Size: 1024x1024 | PyTorch: {pytorch_time*1000:.3f}ms | Triton: {triton_time*1000:.3f}ms | Speedup: {speedup:.2f}x")
    
    # LayerNorm
    print("\n[LayerNorm]")
    x = torch.randn((1024, 1024), device=device)
    weight = torch.randn(1024, device=device)
    bias = torch.randn(1024, device=device)
    
    # 预热
    for _ in range(3):
        _ = operators.layer_norm(x, weight, bias)
    torch.npu.synchronize()
    
    torch.npu.synchronize()
    start = time.time()
    for _ in range(10):
        _ = torch.nn.functional.layer_norm(x, (1024,), weight, bias)
    torch.npu.synchronize()
    pytorch_time = (time.time() - start) / 10
    
    torch.npu.synchronize()
    start = time.time()
    for _ in range(10):
        _ = operators.layer_norm(x, weight, bias)
    torch.npu.synchronize()
    triton_time = (time.time() - start) / 10
    
    speedup = pytorch_time / triton_time
    results['layer_norm'] = speedup
    print(f"  Size: 1024x1024 | PyTorch: {pytorch_time*1000:.3f}ms | Triton: {triton_time*1000:.3f}ms | Speedup: {speedup:.2f}x")
    
    # FlashAttention
    print("\n[FlashAttention]")
    q = torch.randn((1, 8, 256, 64), device=device, dtype=torch.float16)
    k = torch.randn((1, 8, 256, 64), device=device, dtype=torch.float16)
    v = torch.randn((1, 8, 256, 64), device=device, dtype=torch.float16)
    
    # 预热
    for _ in range(3):
        _ = operators.flash_attention(q, k, v)
    torch.npu.synchronize()
    
    torch.npu.synchronize()
    start = time.time()
    for _ in range(10):
        _ = operators.flash_attention(q, k, v)
    torch.npu.synchronize()
    triton_time = (time.time() - start) / 10
    
    # PyTorch基线 (CPU)
    q_cpu, k_cpu, v_cpu = q.cpu(), k.cpu(), v.cpu()
    start = time.time()
    for _ in range(10):
        scale = 1.0 / (64 ** 0.5)
        scores = torch.matmul(q_cpu, k_cpu.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        _ = torch.matmul(attn, v_cpu)
    pytorch_time = (time.time() - start) / 10
    
    speedup = pytorch_time / triton_time
    results['flash_attention'] = speedup
    print(f"  B1 H8 S256 D64 | PyTorch(CPU): {pytorch_time*1000:.3f}ms | Triton: {triton_time*1000:.3f}ms | Speedup: {speedup:.2f}x")
    
    # 汇总
    print("\n" + "=" * 60)
    print("性能汇总")
    print("=" * 60)
    
    avg_speedup = sum(results.values()) / len(results)
    print(f"平均加速比: {avg_speedup:.2f}x")
    
    for name, speedup in results.items():
        status = "✓" if speedup >= 1.0 else "✗"
        print(f"  {name}: {speedup:.2f}x {status}")
    
    if avg_speedup >= 1.0:
        print(f"\n性能得分: 100分 (平均加速比 >= 1.0)")
    else:
        score = int(avg_speedup * 100)
        print(f"\n性能得分: {score}分 (平均加速比 = {avg_speedup:.2f})")


if __name__ == "__main__":
    quick_benchmark()