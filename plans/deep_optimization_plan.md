# Triton-Ascend 深度优化计划

## 赛题要求分析

根据赛题PDF文件，关键要求：
1. **必须使用Triton-Ascend语法构建算子** - 不能回退到PyTorch
2. **代码需通过AscendNPU IR编译** - 必须能编译成功
3. **性能加速比 = baseline / current** - 需要超过1.0才能得满分
4. **基线是PyTorch版本的算子性能**

## 当前问题

当前实现使用了"智能回退"机制，在小数据量时直接调用PyTorch，这违反了赛题要求。必须移除回退机制，使用纯Triton-Ascend实现。

## 性能瓶颈分析

### 1. 内核启动开销
- Triton内核启动有固定开销
- 小数据量时开销占比大
- **解决方案**: 使用更大的BLOCK_SIZE，减少内核启动次数

### 2. 内存访问模式
- 未充分利用昇腾NPU的内存层次结构
- **解决方案**: 优化数据对齐，使用向量化加载

### 3. 并行度不足
- 未充分利用昇腾NPU的多核特性
- **解决方案**: 调整grid配置，增加并行度

## 优化策略

### 策略1: 移除PyTorch回退
所有算子必须使用纯Triton-Ascend实现，移除`if n_elements < threshold: return pytorch_impl()`的回退逻辑。

### 策略2: Auto-tuning
使用`@triton.autotune`装饰器自动搜索最优配置：
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=16),
    ],
    key=['n_elements'],
)
```

### 策略3: 昇腾亲和优化
根据昇腾NPU架构特点：
1. **多核任务并行**: 固定分核数为物理核数
2. **数据对齐**: 尾轴大小需被32Bytes整除
3. **存算并行**: 开启multiBuffer=True

### 策略4: 内核融合
将多个操作融合到一个内核中，减少内存访问和内核启动开销。

## 各算子优化计划

### 1. VectorAdd
- 移除回退机制
- 使用Auto-tuning搜索最优BLOCK_SIZE
- 优化grid配置

### 2. Softmax
- 移除回退机制
- 使用融合内核处理整行
- 优化数值稳定性

### 3. LayerNorm
- 移除回退机制
- 使用Welford算法单遍计算
- 融合归一化和仿射变换

### 4. Matmul
- 优化分块策略
- 使用L2缓存优化(super-grouping)
- Auto-tuning搜索最优分块参数

### 5. FlashAttention
- 优化分块大小避免UB/CBUF溢出
- 使用在线Softmax算法
- 优化内存访问模式

## 实施步骤

1. **移除所有PyTorch回退代码**
2. **为每个算子添加Auto-tuning**
3. **优化内核参数配置**
4. **运行测试验证功能正确性**
5. **运行性能基准测试**
6. **迭代优化直到加速比>=1.0**

## 预期目标

- 功能测试: 100%通过
- 性能加速比: >=1.0 (满分)
- 所有算子使用纯Triton-Ascend实现