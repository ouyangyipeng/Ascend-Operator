# 基于Triton-Ascend构建昇腾亲和算子

本项目是2026年全国大学生计算机系统能力大赛编译系统设计赛（华为毕昇杯）的参赛作品。

## 项目简介

使用Triton-Ascend框架设计并构建昇腾亲和的高性能算子，确保算子功能正确且通过误差检验，最大化计算性能。

## 环境要求

- Python 3.9-3.11
- Triton-Ascend 3.2.0
- CANN 8.5.0+
- torch_npu 2.9.0
- PyTorch 2.9.0

## 安装

```bash
# 安装依赖
pip install ninja cmake wheel pybind11

# 安装PyTorch 2.9.0
pip install torch==2.9.0

# 安装torch_npu（需要昇腾环境）
pip install torch_npu==2.9.0

# 安装Triton-Ascend
pip install triton-ascend
```

### 环境变量配置

```bash
# 设置库路径
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.0.1/lib64:/usr/local/Ascend/ascend-toolkit/8.5.0/cann-8.5.0/lib64:$LD_LIBRARY_PATH
```

## 项目结构

```
Ascend-Operator/
├── operators/            # 算子实现
│   ├── vector_add.py     # 向量加法
│   ├── matmul.py         # 矩阵乘法
│   ├── softmax.py        # Softmax
│   ├── flash_attention.py # Flash Attention
│   ├── layer_norm.py     # Layer Normalization
│   ├── rms_norm.py       # RMS Normalization
│   └── reduction.py      # 归约算子
├── tests/                # 测试用例
│   ├── test_operators.py
│   └── benchmark_operators.py
├── docs/                 # 文档
│   ├── DESIGN_DOCUMENT.md
│   └── TEAM_GUIDE.md
└── PROGRESS.md           # 进度记录
```

## 已实现算子

| 算子 | 描述 | 优化技术 |
|------|------|----------|
| vector_add | 向量加法 | 智能回退、大块处理 |
| matmul | 矩阵乘法 | 分块计算、L2缓存优化 |
| softmax | Softmax归一化 | 融合内核、智能回退 |
| flash_attention | Flash Attention | 在线Softmax、内存优化 |
| layer_norm | Layer Normalization | 融合内核、智能回退 |
| rms_norm | RMS Normalization | 计算简化 |
| reduction | 归约算子 | 向量化计算 |

## 使用示例

```python
import torch
from operators import matmul, softmax, flash_attention

# 矩阵乘法
a = torch.randn(512, 512, device='npu:0', dtype=torch.float16)
b = torch.randn(512, 512, device='npu:0', dtype=torch.float16)
c = matmul(a, b)

# Softmax
x = torch.randn(128, 512, device='npu:0')
y = softmax(x)

# Flash Attention
q = torch.randn(1, 8, 512, 64, device='npu:0', dtype=torch.float16)
k = torch.randn(1, 8, 512, 64, device='npu:0', dtype=torch.float16)
v = torch.randn(1, 8, 512, 64, device='npu:0', dtype=torch.float16)
output = flash_attention(q, k, v)
```

## 运行测试

```bash
# 设置环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.0.1/lib64:/usr/local/Ascend/ascend-toolkit/8.5.0/cann-8.5.0/lib64:$LD_LIBRARY_PATH

# 运行功能测试
python3 -m pytest tests/test_operators.py -v

# 运行性能基准测试
python3 tests/benchmark_operators.py
```

## 测试结果

测试环境：8卡昇腾910B4 (192核鲲鹏920, 1.5TB内存)

### 功能测试
| 算子 | 测试状态 | 通过/总数 |
|------|----------|-----------|
| VectorAdd | ✓ 通过 | 6/6 |
| Matmul | ✓ 通过 | 4/4 |
| Softmax | ✓ 通过 | 4/4 |
| FlashAttention | ✓ 通过 | 2/2 |
| **总计** | **100%** | **17/17** |

### 性能数据

当前实现为纯Triton-Ascend实现，符合比赛规则要求。

| 算子 | 优化策略 |
|------|----------|
| vector_add | Auto-tuning (6种配置) |
| matmul | Auto-tuning (5种配置) |
| softmax | 自适应块大小 |
| layer_norm | Auto-tuning (5种配置) |
| flash_attention | 自适应块大小 |

## 优化策略

### 1. Auto-tuning
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        # ...更多配置
    ],
    key=['n_elements'],
)
@triton.jit
def kernel(...):
    ...
```

### 2. 融合内核
- Softmax：单块处理整行，减少内存访问
- LayerNorm：融合归一化和仿射变换

### 3. 分块计算
- Flash Attention：分块处理避免O(N²)内存
- Matmul：分块提高缓存命中率

### 4. 自适应配置
- 根据输入大小动态选择最优块大小
- 小序列使用小块增加并行度
- 大序列使用大块减少内存访问

## 文档

- [设计文档](docs/DESIGN_DOCUMENT.md) - 详细的算子设计和优化策略
- [团队指南](docs/TEAM_GUIDE.md) - 比赛规则和环境配置指南
- [进度记录](PROGRESS.md) - 开发进度和测试结果

## 参考资料

- [Triton-Ascend用户指南](https://gitcode.com/Ascend/triton-ascend)
- [AscendNPU-IR开源代码仓](https://gitcode.com/Ascend/ascendnpu-ir)
- [CANN社区版本下载](https://www.hiascend.com/developer/download/community/result?module=cann)
- [AscendNPU-IR用户指南](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/AscendNPUIR/ir_001.html)

## 许可证

MIT License