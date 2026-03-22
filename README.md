# 基于Triton-Ascend构建昇腾亲和算子

本项目是2026年全国大学生计算机系统能力大赛编译系统设计赛（华为毕昇杯）的参赛作品。

## 项目简介

使用Triton-Ascend框架设计并构建昇腾亲和的高性能算子，确保算子功能正确且通过误差检验，最大化计算性能。

## 环境要求

- Python 3.9-3.11
- Triton-Ascend 3.2.0
- CANN 8.5.0
- torch_npu 2.7.1

## 安装

```bash
# 安装依赖
pip install ninja cmake wheel pybind11

# 安装Triton-Ascend
pip install triton-ascend

# 安装torch_npu（需要昇腾环境）
pip install torch_npu==2.7.1
```

## 项目结构

```
Ascend-Oper/
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
│   └── DESIGN_DOCUMENT.md
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
# 运行所有测试
python -m pytest tests/test_operators.py -v

# 运行性能基准测试
python tests/test_operators.py
```

## 优化策略

### 1. 多核任务并行
- 将分核数量固定为硬件物理核数
- 使用跨步分配策略均匀分配任务

### 2. 单核数据搬运
- 设置合适的BLOCK_SIZE
- 保证数据对齐
- 开启存算并行

### 3. Auto-tuning
- 自动搜索最优配置
- 根据输入大小选择最佳参数

## 文档

- [设计文档](docs/DESIGN_DOCUMENT.md)
- [进度记录](PROGRESS.md)

## 参考资料

- [Triton-Ascend官方文档](https://gitcode.com/Ascend/triton-ascend)
- [AscendNPU IR用户指南](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/AscendNPUIR/ir_001.html)
- [昇腾CANN文档](https://www.hiascend.com/cann/download)

## 许可证

MIT License

---

*2026年全国大学生计算机系统能力大赛编译系统设计赛参赛作品*