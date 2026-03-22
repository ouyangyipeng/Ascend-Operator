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
- [x] 功能测试（CPU模式）- 所有算子测试通过

#### 进行中
- [ ] 在NPU环境测试算子（需要昇腾硬件）

#### 待办事项
- [ ] 性能基准测试
- [ ] Auto-tuning调优
- [ ] 准备提交材料
- [ ] 录制介绍视频

---

## 项目结构

```
Ascend-Oper/
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
├── tests/                # 测试用例
│   └── test_operators.py
├── docs/                 # 文档
│   └── DESIGN_DOCUMENT.md
├── triton-ascend/        # Triton-Ascend源码
└── ascendnpu-ir/         # AscendNPU IR源码
```

---

## 测试结果

### CPU模式测试（2026-03-22）

所有算子在CPU模式下测试通过：

| 算子 | 测试状态 | 最大误差 |
|------|----------|----------|
| vector_add | ✓ 通过 | 0.000000 |
| matmul | ✓ 通过 | 0.000000 |
| softmax | ✓ 通过 | 0.000000 |
| layer_norm | ✓ 通过 | 0.000000 |
| rms_norm | ✓ 通过 | 0.000000 |
| flash_attention | ✓ 通过 | 0.000000 |

---

## 技术笔记

### 环境信息
- **当前开发环境**：Ubuntu 22.04.5 LTS (WSL2), x86_64
- **目标运行环境**：鲲鹏920 ARM + 昇腾A2/A3 NPU, openEuler
- **Python版本**：3.10.12
- **Triton-Ascend版本**：3.2.0
- **配套CANN版本**：8.5.0

### 代码设计特点

1. **无NPU环境兼容**：所有算子在没有NPU驱动时自动回退到PyTorch实现
2. **延迟导入**：使用`__getattr__`实现延迟导入，避免导入时错误
3. **参考实现**：每个算子都提供PyTorch参考实现用于验证

### Triton-Ascend关键优化技术
1. **多核任务并行**：
   - 将分核数量固定为硬件物理核数
   - 纯Vector算子：分核数=Vector核数量
   - CV融合算子：分核数=Cube核数量

2. **单核数据搬运**：
   - 设置合适的BLOCK_SIZE
   - 保证Tensor尾轴大小数据对齐
   - 存算并行（multiBuffer=True默认开启）

3. **Auto-tuning**：
   - 使用`@triton.autotune`装饰器自动搜索最优配置

### 编译器问题处理
- 如果遇到精度问题，可以尝试调试核内同步选项 `inject_barrier_all=[True|False]`

---

## 未来规划
1. ~~完成环境搭建~~ ✓
2. ~~实现基础算子~~ ✓
3. ~~功能测试~~ ✓
4. 在NPU环境测试算子（需要硬件）
5. 性能优化和Auto-tuning
6. 准备提交材料
7. 录制介绍视频