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
# 设置库路径（使用CANN 8.0.1运行时库 + CANN 8.5.0编译器）
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
│   └── test_operators.py
├── docs/                 # 文档
│   ├── DESIGN_DOCUMENT.md
│   └── TEAM_GUIDE.md
└── PROGRESS.md           # 进度记录
```

## 已实现算子

| 算子 | 描述 | 优化技术 |
|------|------|----------|
| vector_add | 向量加法 | 多核并行、Auto-tuning |
| matmul | 矩阵乘法 | 分块计算、L2缓存优化 |
| softmax | Softmax归一化 | 数值稳定性、行级并行 |
| flash_attention | Flash Attention | 在线Softmax、内存优化 |
| layer_norm | Layer Normalization | Welford算法 |
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

# 运行测试
python3 -m pytest tests/test_operators.py -v
```

## NPU测试结果

测试环境：8卡昇腾910B4 (192核鲲鹏920, 1.5TB内存)

| 算子 | 测试状态 | 通过/总数 | 备注 |
|------|----------|-----------|------|
| VectorAdd | ✓ 通过 | 6/6 | 所有尺寸测试通过 |
| Matmul | ✓ 通过 | 4/4 | FP16精度在可接受范围 |
| Softmax | ✓ 通过 | 4/4 | 所有尺寸测试通过 |
| FlashAttention | ⚠ 部分通过 | 1/2 | 一个用例有数值稳定性问题 |
| **总计** | **94.1%** | **16/17** | - |

### 性能数据

| 算子 | 数据规模 | 平均时间 | 带宽/吞吐量 |
|------|----------|----------|-------------|
| vector_add | 1M elements (4MB) | 0.077 ms | 1.52 GB/s |
| softmax | 1024×1024 | 0.075 ms | - |
| layernorm | 1024×1024 | 0.085 ms | - |

## 优化策略

### 1. 多核任务并行
```python
# 获取物理核数
from operators import get_num_cores
num_cores = get_num_cores()  # 192核

# 将分核数量固定为硬件物理核数
grid = (num_cores,)
```

### 2. 单核数据搬运
- BLOCK_SIZE选择：根据数据类型和缓存大小选择
- 数据对齐：确保内存访问对齐
- 存算并行：使用向量化加载和存储

### 3. Auto-tuning
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4),
    ],
    key=['n_elements'],
)
```

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