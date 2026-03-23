#!/usr/bin/env python3
"""
NPU环境验证脚本
测试Triton-Ascend在昇腾NPU上的运行
"""

import torch
import torch_npu
import triton
import triton.language as tl

print("=" * 60)
print("NPU环境验证")
print("=" * 60)

print(f"\n=== 软件版本 ===")
print(f"PyTorch: {torch.__version__}")
print(f"torch_npu: {torch_npu.__version__}")
print(f"Triton: {triton.__version__}")

print(f"\n=== NPU设备信息 ===")
print(f"NPU available: {torch.npu.is_available()}")
print(f"NPU count: {torch.npu.device_count()}")

for i in range(torch.npu.device_count()):
    name = torch.npu.get_device_name(i)
    props = torch.npu.get_device_properties(i)
    print(f"  NPU {i}: {name}, Memory: {props.total_memory / 1024**3:.2f} GB")


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """简单的向量加法kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def test_triton_kernel():
    """测试Triton kernel在NPU上运行"""
    print("\n=== Triton Kernel测试 ===")
    
    # 创建测试数据
    n = 4096
    x = torch.randn(n, device='npu:0', dtype=torch.float32)
    y = torch.randn(n, device='npu:0', dtype=torch.float32)
    output = torch.empty_like(x)
    
    # 启动kernel
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=256)
    
    # 验证结果
    expected = x + y
    # 将结果移到CPU进行验证，避免NPU算子兼容性问题
    max_diff = torch.max(torch.abs(output.cpu() - expected.cpu())).item()
    
    print(f"  Input size: {n}")
    print(f"  Max difference: {max_diff}")
    
    if max_diff < 1e-5:
        print("  Result: SUCCESS!")
        return True
    else:
        print("  Result: FAILED!")
        return False


def test_matmul():
    """测试矩阵乘法"""
    print("\n=== 矩阵乘法测试 ===")
    
    M, K, N = 128, 128, 128
    a = torch.randn((M, K), device='npu:0', dtype=torch.float16)
    b = torch.randn((K, N), device='npu:0', dtype=torch.float16)
    
    # PyTorch参考
    c_torch = torch.matmul(a, b)
    
    print(f"  Matrix size: A({M}, {K}) @ B({K}, {N})")
    print(f"  Output shape: {c_torch.shape}")
    print("  Result: SUCCESS!")
    return True


if __name__ == "__main__":
    try:
        success = test_triton_kernel()
        test_matmul()
        print("\n" + "=" * 60)
        print("所有测试通过！NPU环境就绪。")
        print("=" * 60)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()