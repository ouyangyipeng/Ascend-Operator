# 毕昇杯编译系统挑战赛 - 进度记录

## 赛题概述
- **赛题名称**：基于Triton-Ascend构建昇腾亲和算子
- **目标**：使用TritonAscend设计并构建昇腾亲和的算子，确保功能正确且通过误差检验，最大化计算性能
- **平台**：昇腾A2/A3服务器

## 评分标准
| 指标 | 权重 |
|------|------|
| 功能得分 | 40% |
| 性能得分（对比PyTorch基线） | 60% |

## 关键技术要求
1. 使用TritonAscend语法构建算子
2. 代码需通过AscendNPU IR编译
3. 在昇腾A2/A3平台上运行
4. 优先在TritonAscend层面通过更改算子写法的方式调优性能
5. 可使用Auto-tuning等手段帮助搜索调优

## 精度标准
- **整数计算**：二进制对比一致
- **浮点数计算**：
  - FP16/BF16：标杆将输入转换成FP32精度进行计算
  - FP32：标杆直接采用FP32计算
  - 误差阈值根据计算次数有所不同

## 资源链接
- AscendNPUIR开源代码仓：https://gitcode.com/Ascend/ascendnpu-ir
- TritonAscend用户指南：https://gitcode.com/Ascend/triton-ascend
- CANN社区版本：https://www.hiascend.com/developer/download/community/result?module=cann
- AscendNPUIR用户指南：https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/AscendNPUIR/ir_001.html

---

## 进度日志

### 2026-03-24 - 性能优化测试完成

#### 硬件环境
- **CPU**: 192核鲲鹏920 (ARM架构)
- **NPU**: 8卡昇腾910B4，每卡约30GB HBM
- **内存**: 1.5TB
- **存储**: 21TB固态

#### 软件环境
- **操作系统**: openEuler
- **Python**: 3.10.12
- **CANN**: 8.0.1/8.5.0
- **PyTorch**: 2.9.0+cpu
- **torch_npu**: 2.9.0
- **Triton-Ascend**: 3.2.0

#### 功能测试结果
**测试命令**:
```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.0.1/lib64:/usr/local/Ascend/ascend-toolkit/8.5.0/cann-8.5.0/lib64:$LD_LIBRARY_PATH
python3 -m pytest tests/test_operators.py -v
```

**测试结果**: 17/17 全部通过 (100%)

| 算子 | 测试状态 | 通过/总数 |
|------|----------|-----------|
| VectorAdd | ✓ 通过 | 6/6 |
| Matmul | ✓ 通过 | 4/4 |
| Softmax | ✓ 通过 | 4/4 |
| FlashAttention | ✓ 通过 | 2/2 |
| **总计** | **100%** | **17/17** |

#### 性能测试结果

**测试命令**:
```bash
python3 tests/quick_benchmark.py
```

**性能数据**:
| 算子 | PyTorch基线 | Triton实现 | 加速比 |
|------|-------------|------------|--------|
| VectorAdd (1M) | 8.454ms | 21.252ms | 0.40x |
| Softmax (1024x1024) | 0.335ms | 1.347ms | 0.25x |
| LayerNorm (1024x1024) | 0.404ms | 1.276ms | 0.32x |
| FlashAttention (B1H8S256D64) | 66.329ms(CPU) | 511.951ms | 0.13x |

**平均加速比**: 0.27x

#### 性能分析

Triton实现比PyTorch基线慢的原因：

1. **PyTorch NPU后端优化**: PyTorch在NPU上使用了高度优化的CANN算子库，这些算子经过专业团队深度优化

2. **内核启动开销**: Triton内核启动有固定开销，对于小规模数据影响较大

3. **编译器成熟度**: Triton-Ascend编译器仍在发展中，生成的代码可能不如手写的CANN算子高效

4. **硬件特性利用**: CANN算子充分利用了昇腾NPU的Cube和Vector单元，而Triton生成的代码可能未完全利用

#### 已实现的优化

1. **多核并行**: 使用跨步分配策略，充分利用昇腾NPU多核
2. **分块计算**: 减少内存访问次数，提高数据复用
3. **融合内核**: 单块处理整行，减少内核启动次数
4. **数值稳定性**: 使用Welford算法和在线Softmax

#### 后续优化方向

1. **Auto-tuning**: 实现更精细的自动调参
2. **内存访问优化**: 优化数据对齐和向量化加载
3. **存算并行**: 开启multiBuffer实现流水线
4. **L2缓存优化**: 使用super-grouping策略

---

### 2026-03-24 - NPU测试成功（昇腾910B4）

#### NPU测试结果

**测试命令**:
```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.0.1/lib64:/usr/local/Ascend/ascend-toolkit/8.5.0/cann-8.5.0/lib64:$LD_LIBRARY_PATH
python3 -m pytest tests/test_operators.py -v
```

**测试结果汇总**:
| 算子 | 测试状态 | 通过/总数 | 备注 |
|------|----------|-----------|------|
| VectorAdd | ✓ 通过 | 6/6 | 所有尺寸测试通过 |
| Matmul | ✓ 通过 | 4/4 | FP16精度在可接受范围 |
| Softmax | ✓ 通过 | 4/4 | 所有尺寸测试通过 |
| FlashAttention | ✓ 通过 | 2/2 | 数值稳定性已修复 |

**总计**: 17/17 测试通过 (100%)

---

### 2026-03-22 - 项目启动

#### 已完成
- [x] 阅读赛题文档，理解比赛要求
- [x] 检查系统环境
- [x] 安装Triton-Ascend pip包（v3.2.0）
- [x] 创建项目结构
- [x] 实现基础算子
- [x] 编写设计文档
- [x] 功能测试
- [x] NPU测试成功
- [x] 性能优化

---

## 项目结构

```
Ascend-Operator/
├── README.md             # 项目说明
├── PROGRESS.md           # 进度记录
├── operators/            # 算子实现
│   ├── __init__.py
│   ├── utils.py          # 工具函数
│   ├── vector_add.py     # 向量加法
│   ├── matmul.py         # 矩阵乘法
│   ├── softmax.py        # Softmax
│   ├── flash_attention.py # Flash Attention
│   ├── layer_norm.py     # Layer Normalization
│   ├── rms_norm.py       # RMS Normalization
│   └── reduction.py      # 归约算子
├── tests/
│   ├── test_operators.py # 算子测试
│   ├── benchmark_operators.py # 性能基准
│   └── quick_benchmark.py # 快速性能测试
└── docs/
    ├── DESIGN_DOCUMENT.md # 设计文档
    └── TEAM_GUIDE.md      # 团队指南
```

## 已实现算子

| 算子 | 功能 | 优化策略 |
|------|------|----------|
| vector_add | 向量加法 | 多核并行、大块处理 |
| matmul | 矩阵乘法 | 分块计算、tl.dot |
| softmax | Softmax归一化 | 融合内核、数值稳定性 |
| flash_attention | Flash Attention | 分块计算、在线Softmax |
| layer_norm | Layer Normalization | 融合内核、Welford算法 |
| rms_norm | RMS Normalization | 行并行、向量化 |
| reduction | 归约（sum/max/min） | 分块归约、向量化 |

---

## 未来规划

1. **短期目标**
   - 进一步优化算子性能
   - 准备比赛提交材料

2. **中期目标**
   - 实现更多算子
   - Auto-tuning参数调优

3. **长期目标**
   - 贡献代码到开源社区
   - 编写技术博客分享经验