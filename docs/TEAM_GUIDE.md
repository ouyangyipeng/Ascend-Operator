# 毕昇杯编译系统挑战赛 - 团队指南

> 本文档面向团队成员，详细介绍比赛赛题、项目状态和后续工作。

---

## 一、赛题内容详细解释

### 1.1 比赛名称
**2026年全国大学生计算机系统能力大赛编译系统设计赛（华为毕昇杯）**

### 1.2 赛题名称
**《基于Triton-Ascend构建昇腾亲和算子》**

### 1.3 赛题要求
参赛队需要使用 **Triton-Ascend** 框架设计并构建昇腾亲和的算子，具体要求如下：

1. **使用TritonAscend语法构建算子**：代码必须使用Triton-Ascend提供的Python API编写
2. **代码需通过AscendNPU IR编译**：最终代码需要能够被AscendNPU IR编译器编译成可在昇腾NPU上运行的可执行文件
3. **在昇腾A2/A3平台上运行**：目标硬件是华为昇腾A2/A3系列NPU
4. **功能正确**：算子必须通过精度检验
5. **性能优化**：在功能正确的前提下，最大化计算性能

### 1.4 比赛提供的资源
- 算子论文：描述算子的算法原理
- PyTorch参考实现：作为功能对照和性能基线
- 测试配置：一系列输入配置用于测试

---

## 二、赛题意义

### 2.1 技术意义
1. **掌握Triton编程框架**：Triton是近年来流行的Python化GPU/NPU编程框架，学习它有助于理解现代并行计算
2. **深入理解昇腾架构**：通过优化算子，深入理解昇腾NPU的硬件特性
3. **编译器技术实践**：将高级语言编译到底层硬件指令的过程

### 2.2 实际应用
- 昇腾NPU广泛应用于AI训练和推理场景
- 高效的算子实现可以显著提升模型性能
- 为国产AI芯片生态做贡献

---

## 三、规则遵守情况

### 3.1 必须遵守的规则（已遵守 ✓）

| 规则 | 状态 | 说明 |
|------|------|------|
| 使用TritonAscend语法 | ✓ 已遵守 | 所有算子使用`@triton.jit`装饰器 |
| 通过AscendNPU IR编译 | ⏳ 待验证 | 需要在昇腾环境测试 |
| 在昇腾A2/A3平台运行 | ⏳ 待验证 | 需要昇腾硬件 |
| 功能正确通过精度检验 | ✓ 已遵守 | CPU模式测试通过 |
| 性能超过PyTorch基线 | ⏳ 待优化 | 需要在NPU上测试性能 |

### 3.2 禁止事项（未违反 ✓）

| 禁止事项 | 状态 | 说明 |
|----------|------|------|
| 代码抄袭（重复率>20%） | ✓ 未违反 | 所有代码为原创实现 |
| 使用他人代码不做说明 | ✓ 未违反 | 引用的开源代码已标注 |
| 修改禁止修改的部分 | ✓ 未违反 | 未修改编译器核心代码 |

### 3.3 必须说明的事项

**引用的开源代码**：
1. **Triton-Ascend** (https://gitcode.com/Ascend/triton-ascend) - MIT许可证
   - 用于Triton语言运行时和编译器后端
2. **AscendNPU-IR** (https://gitcode.com/Ascend/ascendnpu-ir) - Apache 2.0许可证
   - 用于昇腾NPU的中间表示和编译

所有引用已在代码头部和设计文档中标注。

---

## 四、评分方式详解

### 4.1 初赛评分（总分100分）

#### 功能得分（40%）
- **测试方式**：大赛提供一系列算子的输入配置
- **评分标准**：
  - 所有配置都通过：100分
  - 部分通过：按通过比例计分
  - 功能失败（编译失败/运行时错误/精度不达标）：0分

#### 性能得分（60%）
- **基线**：PyTorch版本的算子性能
- **计算公式**：`加速比 = baseline / current`
- **评分标准**：
  - 加速比 ≥ 1.0：100分
  - 加速比 < 1.0：按比例扣分（如加速比0.5则计50分）

### 4.2 决赛评分

| 评分项 | 权重 |
|--------|------|
| 量化评测指标（与初赛一致） | 30% |
| 方案设计文档 | 20% |
| 团队协作及现场答辩 | 50% |

### 4.3 精度标准

| 数据类型 | 计算次数 | 误差阈值 |
|----------|----------|----------|
| FP16 | < 2048 | 2^-8 |
| FP16 | ≥ 2048 | 2^-7 |
| BF16 | < 2048 | 2^-7 |
| BF16 | ≥ 2048 | 2^-8 |
| FP32 | < 2048 | 2^-11 |
| FP32 | 2048-16384 | 2^-10 |
| FP32 | ≥ 16384 | 2^-9 |

---

## 五、测试用例情况

### 5.1 当前测试状态
- **初赛测试用例**：大赛尚未公布具体测试用例
- **当前测试**：我们使用自建的测试用例进行CPU模式测试
- **测试结果**：所有算子在CPU模式下功能正确

### 5.2 测试策略
1. **功能测试**：与PyTorch参考实现对比，确保精度达标
2. **边界测试**：测试不同输入尺寸
3. **性能测试**：在NPU上测量执行时间

### 5.3 后续测试计划
- 等待大赛公布测试用例后进行针对性优化
- 在昇腾NPU上进行实际测试

---

## 六、第三方IP引用情况

### 6.1 引用的开源项目

| 项目 | 用途 | 许可证 | 标注位置 |
|------|------|--------|----------|
| Triton-Ascend | Triton语言运行时 | MIT | 设计文档、README |
| AscendNPU-IR | 昇腾编译器后端 | Apache 2.0 | 设计文档、README |

### 6.2 代码原创性
- 所有算子实现代码为团队原创
- 参考了Triton-Ascend官方教程的编程模式
- 未直接复制任何现有实现

---

## 七、昇腾NPU环境配置指南

### 7.1 硬件要求
- 昇腾A2/A3系列NPU
- 鲲鹏920 ARM CPU（推荐）或x86_64 CPU
- 最小32GB显存

### 7.2 软件安装步骤

#### 步骤1：安装CANN 8.5.0
```bash
# 下载CANN 8.5.0（Triton-Ascend 3.2.0需要8.5.0+）
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-toolkit_8.5.0_linux-aarch64.run

# 安装（使用--quiet自动接受EULA）
chmod +x Ascend-cann-toolkit_8.5.0_linux-aarch64.run
./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --quiet

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/8.5.0/cann-8.5.0/set_env.sh
```

#### 步骤2：安装PyTorch和torch_npu
```bash
# 安装PyTorch 2.9.0
pip install torch==2.9.0

# 安装torch_npu 2.9.0（需要匹配PyTorch版本）
pip install torch_npu==2.9.0
```

#### 步骤3：安装Triton-Ascend
```bash
pip install triton-ascend
```

#### 步骤4：配置环境变量
```bash
# 关键：使用CANN 8.0.1运行时库 + CANN 8.5.0编译器
# 这是因为torch_npu 2.9.0与CANN 8.5.0运行时有兼容性问题
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.0.1/lib64:/usr/local/Ascend/ascend-toolkit/8.5.0/cann-8.5.0/lib64:$LD_LIBRARY_PATH
```

#### 步骤5：克隆项目
```bash
git clone git@github.com:ouyangyipeng/Ascend-Operator.git
cd Ascend-Operator
```

#### 步骤6：运行测试
```bash
# 设置环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.0.1/lib64:/usr/local/Ascend/ascend-toolkit/8.5.0/cann-8.5.0/lib64:$LD_LIBRARY_PATH

# 运行完整测试套件
python3 -m pytest tests/test_operators.py -v
```

### 7.3 获取昇腾NPU资源
1. **华为云ModelArts**：提供昇腾NPU云服务
2. **启智社区**：https://openi.pcl.ac.cn 提供免费的昇腾算力
3. **学校实验室**：如果有昇腾服务器

---

## 八、当前进度

### 8.1 已完成工作

| 任务 | 状态 | 完成度 |
|------|------|--------|
| 环境搭建 | ✓ 完成 | 100% |
| 算子实现 | ✓ 完成 | 100% |
| CPU功能测试 | ✓ 完成 | 100% |
| 设计文档 | ✓ 完成 | 100% |
| NPU测试 | ✓ 完成 | 94.1% (16/17) |
| 性能优化 | ⏳ 待进行 | 0% |
| 提交材料 | ⏳ 待进行 | 0% |

### 8.2 已实现的算子

| 算子 | 功能 | CPU测试 | NPU测试 |
|------|------|---------|---------|
| vector_add | 向量加法 | ✓ 通过 | ✓ 6/6 |
| matmul | 矩阵乘法 | ✓ 通过 | ✓ 4/4 |
| softmax | Softmax归一化 | ✓ 通过 | ✓ 4/4 |
| flash_attention | Flash Attention | ✓ 通过 | ⚠ 1/2 |
| layer_norm | Layer Normalization | ✓ 通过 | ✓ 通过 |
| rms_norm | RMS Normalization | ✓ 通过 | 待测试 |
| reduction | 归约算子 | ✓ 通过 | 待测试 |

### 8.3 NPU测试环境

- **硬件**: 8卡昇腾910B4, 192核鲲鹏920, 1.5TB内存
- **软件**: CANN 8.0.1/8.5.0, PyTorch 2.9.0, torch_npu 2.9.0, Triton-Ascend 3.2.0
- **测试结果**: 16/17 测试通过 (94.1%)

### 8.3 待完成工作
1. **在昇腾NPU上测试**：验证算子能否正确编译和运行
2. **性能优化**：使用Auto-tuning等技术提升性能
3. **准备提交材料**：
   - 源码打包
   - 设计文档PDF
   - 5分钟介绍视频

---

## 九、优化技术说明

### 9.1 已使用的优化技术

#### 1. 多核任务并行
```python
# 将分核数量固定为硬件物理核数
for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
    # 处理任务
```
- 针对昇腾NPU的AI Core数量进行优化
- 避免过多任务导致核启动开销

#### 2. 分块计算（Tiling）
```python
# 矩阵乘法分块
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 32
```
- 减少内存访问次数
- 提高数据复用率

#### 3. 数值稳定性优化
```python
# Softmax减去最大值防止溢出
x_exp = tl.exp(x - max_val)
```

#### 4. 在线算法
- Flash Attention使用在线Softmax
- Layer Norm使用Welford算法

### 9.2 待实施的优化

| 优化技术 | 说明 | 优先级 |
|----------|------|--------|
| Auto-tuning | 自动搜索最优配置 | 高 |
| 数据对齐 | 保证尾轴对齐 | 高 |
| 存算并行 | 开启multiBuffer | 中 |
| L2缓存优化 | Super-grouping | 中 |
| 算子融合 | 融合多个算子 | 低 |

---

## 十、项目文件说明

```
Ascend-Oper/
├── README.md                 # 项目说明
├── PROGRESS.md               # 进度记录
├── operators/                # 算子实现
│   ├── __init__.py          # 模块入口
│   ├── utils.py             # 工具函数
│   ├── vector_add.py        # 向量加法
│   ├── matmul.py            # 矩阵乘法
│   ├── softmax.py           # Softmax
│   ├── flash_attention.py   # Flash Attention
│   ├── layer_norm.py        # Layer Normalization
│   ├── rms_norm.py          # RMS Normalization
│   └── reduction.py         # 归约算子
├── tests/                    # 测试用例
│   └── test_operators.py
├── docs/                     # 文档
│   ├── TEAM_GUIDE.md        # 团队指南（本文档）
│   └── DESIGN_DOCUMENT.md   # 设计文档
├── triton-ascend/           # Triton-Ascend源码
└── ascendnpu-ir/            # AscendNPU IR源码
```

---

## 十一、常见问题

### Q1: 为什么CPU测试通过但NPU可能失败？
A: CPU和NPU的硬件架构不同，某些操作在NPU上可能有不同的行为。需要在实际NPU上验证。

### Q2: 如何判断性能是否达标？
A: 与PyTorch基线对比，加速比≥1.0即为达标。

### Q3: 比赛什么时候公布测试用例？
A: 通常在比赛正式开始后公布，请关注大赛官网：https://compiler.educg.net

### Q4: 遇到编译器问题怎么办？
A: 可以尝试调整`inject_barrier_all`选项，或记录问题反馈给组委会。

---

## 十二、联系方式

- **大赛官网**：https://compiler.educg.net
- **Triton-Ascend仓库**：https://gitcode.com/Ascend/triton-ascend
- **问题反馈**：通过大赛官方渠道

---

*文档版本：v1.0*
*最后更新：2026-03-22*