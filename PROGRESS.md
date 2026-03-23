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

### 2026-03-24 - NPU测试成功（昇腾910B4）

#### 硬件环境
- **CPU**: 192核鲲鹏920 (ARM架构)
- **NPU**: 8卡昇腾910B4，每卡约30GB HBM
- **内存**: 1.5TB
- **存储**: 21TB固态

#### 软件环境（最终配置）
- **操作系统**: openEuler
- **Python**: 3.10.12
- **CANN**: 8.5.0 + 8.0.1运行时库
- **PyTorch**: 2.9.0
- **torch_npu**: 2.9.0
- **Triton-Ascend**: 3.2.0

#### 环境配置关键步骤

1. **安装CANN 8.5.0**（提供bishengir-compile编译器）
   ```bash
   wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-toolkit_8.5.0_linux-aarch64.run
   chmod +x Ascend-cann-toolkit_8.5.0_linux-aarch64.run
   ./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --quiet
   ```

2. **安装PyTorch 2.9.0 + torch_npu 2.9.0**
   ```bash
   pip install torch==2.9.0
   pip install torch_npu==2.9.0
   ```

3. **配置环境变量**
   ```bash
   export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.0.1/lib64:/usr/local/Ascend/ascend-toolkit/8.5.0/cann-8.5.0/lib64:$LD_LIBRARY_PATH
   ```

4. **创建必要的符号链接**
   ```bash
   # bishengir-compile编译器
   ln -sf /usr/local/Ascend/ascend-toolkit/8.5.0/cann-8.5.0/tools/bishengir/bin/bishengir-compile /usr/local/bin/
   ln -sf /usr/local/Ascend/ascend-toolkit/8.5.0/cann-8.5.0/tools/bishengir/bin/hivmc /usr/local/bin/
   
   # bisheng编译器（替换CANN 8.0.1版本）
   ln -sf /usr/local/Ascend/ascend-toolkit/8.5.0/cann-8.5.0/tools/bishengir/bin/bisheng /usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin/
   ```

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
| FlashAttention | ⚠ 部分通过 | 1/2 | 一个测试用例有数值稳定性问题 |

**总计**: 16/17 测试通过 (94.1%)

#### 性能数据

| 算子 | 数据规模 | 平均时间 | 带宽/吞吐量 |
|------|----------|----------|-------------|
| vector_add | 1M elements (4MB) | 0.077 ms | 1.52 GB/s |
| softmax | 1024x1024 | 0.075 ms | - |
| layernorm | 1024x1024 | 0.085 ms | - |

#### 已解决的问题

1. **CANN版本兼容性**
   - Triton-Ascend 3.2.0 需要 CANN 8.5.0+ 的 bishengir-compile
   - 解决方案：安装CANN 8.5.0并配置符号链接

2. **torch_npu运行时兼容性**
   - torch_npu 2.9.0 需要CANN 8.5.0头文件但与8.5.0运行时库不兼容
   - 解决方案：使用CANN 8.0.1运行时库 + CANN 8.5.0编译工具

3. **模块导入问题**
   - Python模块名与函数名冲突导致`TypeError: 'module' object is not callable`
   - 解决方案：修改`operators/__init__.py`使用`__getattr__`缓存机制

4. **Triton语法兼容性**
   - `**`幂运算符不支持，需用乘法代替
   - `tl.dot`要求操作数类型一致

---

### 2026-03-22 - 项目启动

#### 已完成
- [x] 阅读赛题文档，理解比赛要求
- [x] 检查系统环境（Ubuntu 22.04.5 LTS, Python 3.10.12, Git 2.34.1, CMake 4.2.3）
- [x] 克隆TritonAscend仓库
- [x] 克隆AscendNPUIR仓库
- [x] 安装Python依赖（ninja, cmake, wheel, pybind11）
- [x] 安装Triton-Ascend pip包（v3.2.0）
- [x] 创建项目结构（operators/, tests/, docs/）
- [x] 实现基础算子：
  - vector_add: 向量加法 ✓
  - matmul: 矩阵乘法 ✓
  - softmax: Softmax归一化 ✓
  - flash_attention: Flash Attention ✓
  - layer_norm: Layer Normalization ✓
  - rms_norm: RMS Normalization ✓
  - reduction: 归约算子（sum, max, min） ✓
- [x] 编写设计文档（docs/DESIGN_DOCUMENT.md）
- [x] 编写README.md
- [x] 编写团队指南（docs/TEAM_GUIDE.md）
- [x] 功能测试（CPU模式）- 所有算子测试通过
- [x] **NPU测试成功** - 16/17测试通过
- [x] 推送代码到GitHub（https://github.com/ouyangyipeng/Ascend-Operator）

#### 待优化
- [ ] FlashAttention数值稳定性优化
- [ ] 性能基准测试
- [ ] Auto-tuning调优
- [ ] 准备提交材料

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
│   └── test_operators.py # 算子测试
└── docs/
    ├── DESIGN_DOCUMENT.md # 设计文档
    └── TEAM_GUIDE.md      # 团队指南
```

## 已实现算子

| 算子 | 功能 | 优化策略 |
|------|------|----------|
| vector_add | 向量加法 | 多核并行、Auto-tuning |
| matmul | 矩阵乘法 | 分块计算、向量化加载 |
| softmax | Softmax归一化 | 行并行、数值稳定性 |
| flash_attention | Flash Attention | 分块计算、在线Softmax |
| layer_norm | Layer Normalization | 行并行、Welford算法 |
| rms_norm | RMS Normalization | 行并行、向量化 |
| reduction | 归约（sum/max/min） | 树形归约、多核并行 |

---

## 测试记录

### CPU模式测试（2026-03-22）
所有算子在CPU模式下测试通过。

### NPU模式测试（2026-03-24）
- **测试环境**: 8卡昇腾910B4
- **测试结果**: 16/17 通过 (94.1%)
- **失败测试**: FlashAttention test_basic[1-8-128-64] (数值稳定性问题)

---

## 未来规划

1. **短期目标**
   - 修复FlashAttention数值稳定性问题
   - 完成性能基准测试
   - 优化算子性能

2. **中期目标**
   - 实现更多算子
   - Auto-tuning参数调优
   - 准备比赛提交材料

3. **长期目标**
   - 贡献代码到开源社区
   - 编写技术博客分享经验