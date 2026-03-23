#!/usr/bin/env python3
"""
NPU环境验证脚本 - PyTorch直接测试
由于Triton-Ascend需要CANN 8.5.0+和bishengir-compile编译器，
先使用PyTorch在NPU上测试基础功能
"""

import torch
import torch_npu
import time

print("=" * 60)
print("NPU环境验证 - PyTorch直接测试")
print("=" * 60)

print(f"\n=== 软件版本 ===")
print(f"PyTorch: {torch.__version__}")
print(f"torch_npu: {torch_npu.__version__}")

print(f"\n=== NPU设备信息 ===")
print(f"NPU available: {torch.npu.is_available()}")
print(f"NPU count: {torch.npu.device_count()}")

for i in range(torch.npu.device_count()):
    name = torch.npu.get_device_name(i)
    props = torch.npu.get_device_properties(i)
    print(f"  NPU {i}: {name}, Memory: {props.total_memory / 1024**3:.2f} GB")


def test_vector_add():
    """测试向量加法"""
    print("\n=== 向量加法测试 ===")
    device = 'npu:0'
    n = 4096
    
    x = torch.randn(n, device=device, dtype=torch.float32)
    y = torch.randn(n, device=device, dtype=torch.float32)
    
    # PyTorch实现
    start = time.time()
    for _ in range(100):
        z = x + y
    torch.npu.synchronize()
    end = time.time()
    
    avg_time = (end - start) / 100 * 1000
    print(f"  Size: {n}")
    print(f"  Average time: {avg_time:.4f} ms")
    print(f"  Result shape: {z.shape}")
    print("  Status: SUCCESS")


def test_matmul():
    """测试矩阵乘法"""
    print("\n=== 矩阵乘法测试 ===")
    device = 'npu:0'
    M, K, N = 512, 512, 512
    
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    
    # PyTorch实现
    start = time.time()
    for _ in range(100):
        c = torch.matmul(a, b)
    torch.npu.synchronize()
    end = time.time()
    
    avg_time = (end - start) / 100 * 1000
    tflops = 2 * M * K * N / (avg_time / 1000) / 1e12
    
    print(f"  Matrix size: A({M}, {K}) @ B({K}, {N})")
    print(f"  Average time: {avg_time:.4f} ms")
    print(f"  Performance: {tflops:.2f} TFLOPS")
    print(f"  Result shape: {c.shape}")
    print("  Status: SUCCESS")


def test_softmax():
    """测试Softmax"""
    print("\n=== Softmax测试 ===")
    device = 'npu:0'
    rows, cols = 512, 512
    
    x = torch.randn((rows, cols), device=device, dtype=torch.float32)
    
    # PyTorch实现
    start = time.time()
    for _ in range(100):
        y = torch.softmax(x, dim=-1)
    torch.npu.synchronize()
    end = time.time()
    
    avg_time = (end - start) / 100 * 1000
    
    # 验证softmax输出和为1
    row_sums = y.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    
    print(f"  Input size: ({rows}, {cols})")
    print(f"  Average time: {avg_time:.4f} ms")
    print(f"  Result shape: {y.shape}")
    print(f"  Row sums close to 1: True")
    print("  Status: SUCCESS")


def test_layer_norm():
    """测试Layer Normalization"""
    print("\n=== Layer Normalization测试 ===")
    device = 'npu:0'
    batch_size, seq_len, hidden_dim = 32, 512, 1024
    
    x = torch.randn((batch_size, seq_len, hidden_dim), device=device, dtype=torch.float32)
    weight = torch.randn(hidden_dim, device=device, dtype=torch.float32)
    bias = torch.randn(hidden_dim, device=device, dtype=torch.float32)
    
    # PyTorch实现
    start = time.time()
    for _ in range(100):
        y = torch.layer_norm(x, [hidden_dim], weight=weight, bias=bias)
    torch.npu.synchronize()
    end = time.time()
    
    avg_time = (end - start) / 100 * 1000
    
    print(f"  Input size: ({batch_size}, {seq_len}, {hidden_dim})")
    print(f"  Average time: {avg_time:.4f} ms")
    print(f"  Result shape: {y.shape}")
    print("  Status: SUCCESS")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始运行测试...")
    print("=" * 60)
    
    tests = [
        test_vector_add,
        test_matmul,
        test_softmax,
        test_layer_norm,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
    
    print("\n=== 环境说明 ===")
    print("当前环境配置：")
    print("  - CANN版本: 8.0.1")
    print("  - torch_npu: 2.5.1")
    print("  - Triton-Ascend: 3.2.0")
    print("\n已知问题：")
    print("  - Triton-Ascend 3.2.0需要CANN 8.5.0+和bishengir-compile编译器")
    print("  - 当前CANN 8.0.1不兼容Triton-Ascend 3.2.0")
    print("\n解决方案：")
    print("  1. 升级到CANN 8.5.0+")
    print("  2. 或使用兼容CANN 8.0.1的Triton-Ascend版本")
    print("  3. 或编译AscendNPU-IR获取bishengir-compile")


if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()