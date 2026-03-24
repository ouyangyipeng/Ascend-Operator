#!/usr/bin/env python3
"""
性能基准测试脚本
对比Triton实现与PyTorch基线的性能
"""

import torch
import torch_npu
import time
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import operators


def set_device():
    """设置NPU设备"""
    if torch.npu.is_available():
        torch.npu.set_device(0)
        print(f"使用NPU: {torch.npu.get_device_name(0)}")
        return 'npu:0'
    return 'cpu'


def benchmark_vector_add():
    """向量加法性能测试"""
    print("\n" + "=" * 60)
    print("VectorAdd 性能测试")
    print("=" * 60)
    
    device = set_device()
    sizes = [1024, 10240, 102400, 1048576, 10485760]  # 1K, 10K, 100K, 1M, 10M
    
    results = []
    
    for size in sizes:
        x = torch.randn(size, device=device)
        y = torch.randn(size, device=device)
        
        # PyTorch基线
        torch.npu.synchronize()
        start = time.time()
        for _ in range(100):
            _ = x + y
        torch.npu.synchronize()
        pytorch_time = (time.time() - start) / 100
        
        # Triton实现
        torch.npu.synchronize()
        start = time.time()
        for _ in range(100):
            _ = operators.vector_add(x, y)
        torch.npu.synchronize()
        triton_time = (time.time() - start) / 100
        
        speedup = pytorch_time / triton_time
        results.append((size, pytorch_time, triton_time, speedup))
        
        print(f"Size: {size:10d} | PyTorch: {pytorch_time*1000:8.3f}ms | Triton: {triton_time*1000:8.3f}ms | Speedup: {speedup:6.2f}x")
    
    return results


def benchmark_matmul():
    """矩阵乘法性能测试"""
    print("\n" + "=" * 60)
    print("Matmul 性能测试")
    print("=" * 60)
    
    device = set_device()
    configs = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]
    
    results = []
    
    for M, K, N in configs:
        a = torch.randn((M, K), device=device, dtype=torch.float16)
        b = torch.randn((K, N), device=device, dtype=torch.float16)
        
        # PyTorch基线
        torch.npu.synchronize()
        start = time.time()
        for _ in range(100):
            _ = torch.matmul(a, b)
        torch.npu.synchronize()
        pytorch_time = (time.time() - start) / 100
        
        # Triton实现
        torch.npu.synchronize()
        start = time.time()
        for _ in range(100):
            _ = operators.matmul(a, b)
        torch.npu.synchronize()
        triton_time = (time.time() - start) / 100
        
        speedup = pytorch_time / triton_time
        tflops = 2 * M * K * N / triton_time / 1e12
        
        results.append((M, K, N, pytorch_time, triton_time, speedup, tflops))
        
        print(f"Shape: ({M},{K})x({K},{N}) | PyTorch: {pytorch_time*1000:8.3f}ms | Triton: {triton_time*1000:8.3f}ms | Speedup: {speedup:6.2f}x | TFLOPS: {tflops:6.2f}")
    
    return results


def benchmark_softmax():
    """Softmax性能测试"""
    print("\n" + "=" * 60)
    print("Softmax 性能测试")
    print("=" * 60)
    
    device = set_device()
    configs = [
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ]
    
    results = []
    
    for rows, cols in configs:
        x = torch.randn((rows, cols), device=device)
        
        # PyTorch基线
        torch.npu.synchronize()
        start = time.time()
        for _ in range(100):
            _ = torch.softmax(x, dim=-1)
        torch.npu.synchronize()
        pytorch_time = (time.time() - start) / 100
        
        # Triton实现
        torch.npu.synchronize()
        start = time.time()
        for _ in range(100):
            _ = operators.softmax(x)
        torch.npu.synchronize()
        triton_time = (time.time() - start) / 100
        
        speedup = pytorch_time / triton_time
        results.append((rows, cols, pytorch_time, triton_time, speedup))
        
        print(f"Shape: ({rows},{cols}) | PyTorch: {pytorch_time*1000:8.3f}ms | Triton: {triton_time*1000:8.3f}ms | Speedup: {speedup:6.2f}x")
    
    return results


def benchmark_flash_attention():
    """Flash Attention性能测试"""
    print("\n" + "=" * 60)
    print("Flash Attention 性能测试")
    print("=" * 60)
    
    device = set_device()
    configs = [
        (1, 8, 128, 64),
        (2, 8, 256, 64),
        (1, 16, 512, 64),
        (2, 8, 1024, 64),
    ]
    
    results = []
    
    for batch, heads, seq_len, head_dim in configs:
        q = torch.randn((batch, heads, seq_len, head_dim), device=device, dtype=torch.float16)
        k = torch.randn((batch, heads, seq_len, head_dim), device=device, dtype=torch.float16)
        v = torch.randn((batch, heads, seq_len, head_dim), device=device, dtype=torch.float16)
        
        # PyTorch基线
        torch.npu.synchronize()
        start = time.time()
        for _ in range(50):  # Flash Attention较慢，减少迭代次数
            scale = 1.0 / (head_dim ** 0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            _ = torch.matmul(attn, v)
        torch.npu.synchronize()
        pytorch_time = (time.time() - start) / 50
        
        # Triton实现
        torch.npu.synchronize()
        start = time.time()
        for _ in range(50):
            _ = operators.flash_attention(q, k, v)
        torch.npu.synchronize()
        triton_time = (time.time() - start) / 50
        
        speedup = pytorch_time / triton_time
        results.append((batch, heads, seq_len, head_dim, pytorch_time, triton_time, speedup))
        
        print(f"B{batch} H{heads} S{seq_len} D{head_dim} | PyTorch: {pytorch_time*1000:8.3f}ms | Triton: {triton_time*1000:8.3f}ms | Speedup: {speedup:6.2f}x")
    
    return results


def benchmark_layer_norm():
    """LayerNorm性能测试"""
    print("\n" + "=" * 60)
    print("LayerNorm 性能测试")
    print("=" * 60)
    
    device = set_device()
    configs = [
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ]
    
    results = []
    
    for rows, cols in configs:
        x = torch.randn((rows, cols), device=device)
        weight = torch.randn(cols, device=device)
        bias = torch.randn(cols, device=device)
        
        # PyTorch基线
        torch.npu.synchronize()
        start = time.time()
        for _ in range(100):
            _ = torch.nn.functional.layer_norm(x, (cols,), weight, bias)
        torch.npu.synchronize()
        pytorch_time = (time.time() - start) / 100
        
        # Triton实现
        torch.npu.synchronize()
        start = time.time()
        for _ in range(100):
            _ = operators.layer_norm(x, weight, bias)
        torch.npu.synchronize()
        triton_time = (time.time() - start) / 100
        
        speedup = pytorch_time / triton_time
        results.append((rows, cols, pytorch_time, triton_time, speedup))
        
        print(f"Shape: ({rows},{cols}) | PyTorch: {pytorch_time*1000:8.3f}ms | Triton: {triton_time*1000:8.3f}ms | Speedup: {speedup:6.2f}x")
    
    return results


def print_summary(all_results):
    """打印性能汇总"""
    print("\n" + "=" * 60)
    print("性能汇总")
    print("=" * 60)
    
    # 计算平均加速比
    all_speedups = []
    for name, results in all_results.items():
        for result in results:
            speedup = result[-1]  # 最后一个元素是加速比
            if speedup > 0:  # 过滤异常值
                all_speedups.append(speedup)
    
    avg_speedup = sum(all_speedups) / len(all_speedups)
    
    print(f"平均加速比: {avg_speedup:.2f}x")
    
    if avg_speedup >= 1.0:
        print(f"性能得分: 100分 (加速比 >= 1)")
    else:
        score = int(avg_speedup * 100)
        print(f"性能得分: {score}分 (加速比 = {avg_speedup:.2f})")


if __name__ == "__main__":
    # 预热
    print("预热中...")
    x = torch.randn(1000, device='npu:0')
    _ = x + x
    torch.npu.synchronize()
    
    # 运行基准测试
    all_results = {}
    
    try:
        all_results['vector_add'] = benchmark_vector_add()
    except Exception as e:
        print(f"VectorAdd测试失败: {e}")
    
    try:
        all_results['matmul'] = benchmark_matmul()
    except Exception as e:
        print(f"Matmul测试失败: {e}")
    
    try:
        all_results['softmax'] = benchmark_softmax()
    except Exception as e:
        print(f"Softmax测试失败: {e}")
    
    try:
        all_results['flash_attention'] = benchmark_flash_attention()
    except Exception as e:
        print(f"Flash Attention测试失败: {e}")
    
    try:
        all_results['layer_norm'] = benchmark_layer_norm()
    except Exception as e:
        print(f"LayerNorm测试失败: {e}")
    
    # 打印汇总
    print_summary(all_results)