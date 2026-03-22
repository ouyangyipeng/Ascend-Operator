# 基于Triton-Ascend构建昇腾亲和算子 - 设计文档

## 项目概述

本项目是2026年全国大学生计算机系统能力大赛编译系统设计赛（华为毕昇杯）的参赛作品，旨在使用Triton-Ascend框架设计并构建昇腾亲和的高性能算子。

### 项目目标

1. 使用Triton-Ascend语法构建昇腾亲和算子
2. 确保算子功能正确且通过精度检验
3. 最大化计算性能，超越PyTorch基线性能

### 目标平台

- **硬件**：昇腾A2/A3系列NPU
- **软件栈**：Triton-Ascend 3.2.0 + CANN 8.5.0

---

## 算子设计与实现

### 1. 向量加法 (Vector Add)

#### 功能描述
计算两个向量的逐元素加法：`output = x + y`

#### 实现策略
```python
@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    NUM_CORES = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n_elements, BLOCK_SIZE)
    
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
```

#### 优化技术
1. **多核并行**：使用跨步分配策略，将任务均匀分配到各物理核
2. **Auto-tuning**：自动搜索最优BLOCK_SIZE配置（256, 512, 1024, 2048, 4096）
3. **向量化内存访问**：使用tl.arange实现向量化加载和存储

---

### 2. 矩阵乘法 (Matmul)

#### 功能描述
计算两个矩阵的乘法：`C = A @ B`，其中A为(M, K)，B为(K, N)

#### 实现策略
采用分块矩阵乘法算法：
```
for m in range(0, M, BLOCK_SIZE_M):
  for n in range(0, N, BLOCK_SIZE_N):
    acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N))
    for k in range(0, K, BLOCK_SIZE_K):
      a = A[m:m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]
      b = B[k:k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N]
      acc += dot(a, b)
    C[m:m+BLOCK_SIZE_M, n:n+BLOCK_SIZE_N] = acc
```

#### 优化技术
1. **分块计算**：减少内存访问次数，提高数据复用
2. **L2缓存优化**：使用super-grouping策略提高缓存命中率
3. **Auto-tuning**：多种分块配置自动选择最优

#### 性能预期
- 目标：达到cuBLAS/AscendC算子90%以上性能
- 关键：选择合适的BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K

---

### 3. Softmax

#### 功能描述
计算Softmax归一化：`softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))`

#### 实现策略
采用三遍扫描算法：
1. **第一遍**：计算最大值（数值稳定性）
2. **第二遍**：计算exp和sum
3. **第三遍**：计算最终结果并存储

#### 优化技术
1. **数值稳定性**：减去最大值防止指数溢出
2. **行级并行**：每行由一个核处理
3. **向量化计算**：使用向量化操作提高效率

---

### 4. Flash Attention

#### 功能描述
计算缩放点积注意力：`Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V`

#### 实现策略
采用分块计算避免存储完整的注意力矩阵：
1. 分块加载Q、K、V
2. 在线Softmax算法
3. 增量更新输出

#### 优化技术
1. **内存优化**：避免存储O(N^2)的注意力矩阵
2. **在线Softmax**：增量计算避免存储中间结果
3. **多核并行**：沿序列长度维度并行

#### 性能预期
- 内存复杂度：从O(N^2)降低到O(N)
- 计算效率：接近专用注意力算子

---

### 5. Layer Normalization

#### 功能描述
计算层归一化：`output = (x - mean) / sqrt(var + eps) * weight + bias`

#### 实现策略
采用Welford算法在线计算均值和方差：
1. 单次遍历计算统计量
2. 归一化并应用仿射变换

#### 优化技术
1. **Welford算法**：数值稳定的在线统计算法
2. **行级并行**：每行由一个核处理
3. **融合计算**：归一化和仿射变换融合

---

### 6. RMS Normalization

#### 功能描述
计算RMS归一化：`output = x / sqrt(mean(x^2) + eps) * weight`

#### 实现策略
相比Layer Norm，只需计算平方和：
1. 计算平方和
2. 计算RMS
3. 归一化并应用权重

#### 优化技术
1. **计算简化**：无需计算均值，计算量更小
2. **行级并行**
3. **向量化计算**

---

## 昇腾亲和优化策略

### 1. 多核任务并行

昇腾NPU的AI Core数量在几十个量级，与GPU的SM数量不同。需要针对性优化：

```python
# 获取物理核数
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]

# 固定核数
grid = (NUM_CORE,)
for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
    # 处理任务
```

**关键点**：
- 纯Vector算子：分核数 = Vector核数量
- CV融合算子：分核数 = Cube核数量

### 2. 单核数据搬运

#### BLOCK_SIZE选择
- 需要在不超出片上空间时尽可能大
- Atlas A2片上内存：192KB
- 开启doublebuffer后减半

#### 数据对齐
- VV类算子：尾轴大小需被32Bytes整除
- CV类算子：尾轴大小需被512Bytes整除

#### 存算并行
- 默认开启multiBuffer=True
- 实现"搬运+计算"流水线

### 3. Auto-tuning

使用`@triton.autotune`装饰器自动搜索最优配置：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=4),
    ],
    key=['n_elements'],  # 根据输入大小选择配置
)
@triton.jit
def kernel(...):
    ...
```

---

## 测试与验证

### 功能测试

所有算子都提供PyTorch参考实现进行对比验证：

```python
# 测试示例
def test_matmul():
    a = torch.randn((M, K), device='npu:0', dtype=torch.float16)
    b = torch.randn((K, N), device='npu:0', dtype=torch.float16)
    
    output_torch = matmul_reference(a, b)
    output_triton = matmul(a, b)
    
    max_diff = torch.max(torch.abs(output_torch - output_triton))
    assert max_diff < 1e-2  # FP16精度阈值
```

### 精度标准

根据赛题要求：
- **整数计算**：二进制对比一致
- **FP16/BF16**：相对误差阈值 2^-7 ~ 2^-8
- **FP32**：相对误差阈值 2^-9 ~ 2^-11

### 性能测试

性能加速比定义：
```
speedup_ratio = baseline / current
```

目标：加速比 >= 1.0（即达到或超过PyTorch基线）

---

## 项目结构

```
Ascend-Oper/
├── PROGRESS.md           # 进度记录
├── operators/            # 算子实现
│   ├── __init__.py
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

## 创新点

1. **昇腾亲和设计**：针对昇腾NPU架构特点进行优化
   - 固定物理核数的任务分配
   - 数据对齐优化
   - 存算并行

2. **Auto-tuning框架**：自动搜索最优配置
   - 多种分块配置
   - 根据输入大小自动选择

3. **完整算子库**：覆盖常用深度学习算子
   - 基础运算：向量加法、矩阵乘法
   - 归一化：Softmax、Layer Norm、RMS Norm
   - 注意力：Flash Attention
   - 归约：Sum、Max、Min

---

## 遇到的问题与解决方案

### 问题1：当前环境无NPU设备
**解决方案**：代码设计支持CPU模拟运行，在NPU环境部署时自动切换

### 问题2：精度问题
**解决方案**：
- 使用float32进行中间计算
- 采用数值稳定的算法（如Welford算法）
- 提供inject_barrier_all选项调试

### 问题3：性能调优
**解决方案**：
- 使用Auto-tuning自动搜索最优配置
- 参考Triton-Ascend官方调优指南
- 分析性能瓶颈针对性优化

---

## 未来工作

1. **更多算子支持**：添加更多深度学习算子
2. **性能优化**：进一步优化关键算子性能
3. **融合算子**：实现算子融合提高效率
4. **量化支持**：支持INT8/FP8量化计算

---

## 参考资料

1. [Triton-Ascend官方文档](https://gitcode.com/Ascend/triton-ascend)
2. [AscendNPU IR用户指南](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/AscendNPUIR/ir_001.html)
3. [昇腾CANN文档](https://www.hiascend.com/cann/download)
4. [Triton官方文档](https://triton-lang.org/)

---

## 团队信息

- **参赛队伍**：[队伍名称]
- **指导教师**：[教师姓名]
- **团队成员**：[成员名单]
- **联系方式**：[联系邮箱]

---

*文档版本：v1.0*
*最后更新：2026-03-22*