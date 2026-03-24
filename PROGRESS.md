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

### 2026-03-24 - 深度优化完成（纯Triton实现）

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

#### 优化成果

**功能测试**: 17/17 全部通过 (100%)

**关键改进**: 移除所有PyTorch回退，实现纯Triton-Ascend实现（符合比赛要求）

**已实现优化**:
1. **Auto-tuning**: vector_add, matmul 使用 `@triton.autotune` 自动搜索最优配置
2. **自适应配置**: softmax, layer_norm, flash_attention 根据输入大小动态选择最优块大小
3. **数值稳定性**: 所有算子添加数值稳定性处理

#### 优化策略

1. **纯Triton实现**
   - 移除所有PyTorch回退机制（符合比赛规则）
   - 仅保留CPU模式用于开发调试

2. **Auto-tuning优化**
   - vector_add: 6种配置 (256-8192 BLOCK_SIZE)
   - matmul: 5种配置 (不同块大小组合)
   - 自动选择最优配置

3. **自适应块大小**
   - softmax: 根据n_cols选择128-2048的块大小
   - layer_norm: 根据N选择最优块大小
   - flash_attention: 根据seq_len和head_dim选择配置

4. **FlashAttention优化**
   - 修复BLOCK_K与head_dim不匹配导致的NaN问题
   - 使用较小的块大小避免UB/CBUF溢出
   - 添加数值稳定性检查

#### 已解决的问题

1. **FlashAttention NaN问题**
   - 原因：BLOCK_K=32但head_dim=64不匹配
   - 解决：调整块大小并添加NaN检查

2. **UB/CBUF溢出**
   - 原因：块大小过大导致缓冲区溢出
   - 解决：减小BLOCK_M和BLOCK_N

3. **Auto-tuning参数冲突**
   - 原因：在调用内核时传递了autotune已定义的参数
   - 解决：使用lambda grid函数或移除显式参数传递

4. **Softmax精度问题**
   - 原因：BLOCK_SIZE小于n_cols时融合内核处理不正确
   - 解决：根据n_cols动态选择融合内核或分块内核

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
│   └── benchmark_operators.py # 性能基准
└── docs/
    ├── DESIGN_DOCUMENT.md # 设计文档
    └── TEAM_GUIDE.md      # 团队指南
```

## 已实现算子

| 算子 | 功能 | 优化策略 |
|------|------|----------|
| vector_add | 向量加法 | 智能回退、大块处理 |
| matmul | 矩阵乘法 | 分块计算、向量化加载 |
| softmax | Softmax归一化 | 融合内核、智能回退 |
| flash_attention | Flash Attention | 分块计算、在线Softmax |
| layer_norm | Layer Normalization | 融合内核、智能回退 |
| rms_norm | RMS Normalization | 行并行、向量化 |
| reduction | 归约（sum/max/min） | 树形归约、多核并行 |

---

## 未来规划

1. **短期目标**
   - 等待Triton-Ascend编译器优化
   - 进一步优化算子性能
   - 准备比赛提交材料

2. **中期目标**
   - 实现更多算子
   - Auto-tuning参数调优

3. **长期目标**
   - 贡献代码到开源社区
   - 编写技术博客分享经验