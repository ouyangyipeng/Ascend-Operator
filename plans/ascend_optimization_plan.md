# Ascend 910B4 深度优化计划

## 一、当前性能基线

| 算子 | 当前加速比 | 目标加速比 | 差距 |
|------|-----------|-----------|------|
| VectorAdd | 0.40x | >0.95x | 需提升2.4x |
| Softmax | 0.25x | >0.95x | 需提升3.8x |
| LayerNorm | 0.32x | >0.95x | 需提升3.0x |
| FlashAttention | 0.13x | >0.95x | 需提升7.3x |

## 二、性能瓶颈根因分析

### 2.1 核心问题：Triton-Ascend vs PyTorch NPU后端

**关键发现**：PyTorch的NPU后端（torch_npu）使用了高度优化的CANN算子库，这些算子经过华为深度优化。而Triton-Ascend生成的内核需要经过以下路径：

```
Triton代码 → Triton-Ascend编译器 → AscendNPU IR → CANN编译器 → NPU二进制
```

相比之下，PyTorch NPU后端直接调用预编译的高效CANN算子。

### 2.2 具体瓶颈分析

#### VectorAdd (0.40x)
- **问题**：内核启动开销大，数据搬运时间占比高
- **根因**：简单内存绑定操作，PyTorch有极致优化

#### Softmax (0.25x)
- **问题**：多次内存访问（3遍扫描：max、exp、normalize）
- **根因**：未充分利用昇腾的向量化单元

#### LayerNorm (0.32x)
- **问题**：两遍扫描（统计量计算 + 归一化）
- **根因**：Welford算法虽然减少了一遍，但仍有优化空间

#### FlashAttention (0.13x)
- **问题**：复杂的分块计算，大量中间结果
- **根因**：内存访问模式不优，未充分利用Cube单元

## 三、Ascend 910B4 硬件特性

### 3.1 核心架构参数
| 组件 | 参数 | 优化启示 |
|------|------|----------|
| AI Core数量 | 32个/芯片 | 多核并行是关键 |
| Cube单元 | 矩阵计算加速 | Matmul/Attention需利用 |
| Vector单元 | 向量计算加速 | 适合Softmax/LayerNorm |
| Unified Buffer | ~1MB/Core | 数据复用是关键 |
| L2 Cache | 共享缓存 | 跨核数据共享 |

### 3.2 内存层次
```
HBM (高带宽内存) → L2 Cache → Unified Buffer → Vector/Cube单元
     ↓                ↓              ↓
  带宽: 1.2TB/s    共享         每核独享
```

### 3.3 关键优化参数
- **数据对齐**：尾轴大小需被32Bytes整除
- **BLOCK_SIZE**：推荐128, 256, 512, 1024, 2048
- **num_stages**：控制流水线深度
- **num_warps**：控制并行度

## 四、Triton-Ascend 特定优化技术

### 4.1 多Buffer优化
```python
# 启用存算并行
@triton.jit
def kernel(..., multiBuffer: tl.constexpr = True):
    # 编译器会自动优化数据搬运和计算的重叠
```

### 4.2 数据对齐优化
```python
# 确保BLOCK_SIZE是32的倍数（对于float32）
BLOCK_SIZE = 128  # 128 * 4 = 512 bytes，满足32字节对齐
```

### 4.3 Super-Grouping（L2缓存优化）
```python
# 对于矩阵乘法，使用super-grouping优化L2缓存
# 让同一列的block共享K的数据
pid = tl.program_id(0)
num_pid_m = tl.cdiv(M, BLOCK_M)
pid_m = pid // tl.cdiv(N, BLOCK_N)
pid_n = pid % tl.cdiv(N, BLOCK_N)
```

### 4.4 Auto-tuning策略
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=16),
    ],
    key=['n_elements'],  # 根据输入大小选择配置
)
```

## 五、各算子优化方案

### 5.1 VectorAdd优化

**当前问题**：
- 简单操作，内核启动开销占比大
- PyTorch NPU后端有极致优化

**优化策略**：
1. **增大BLOCK_SIZE**：减少内核启动次数
2. **多核并行优化**：使用跨步分配充分利用32个AI Core
3. **向量化加载**：使用tl.load的向量化特性

**具体实现**：
```python
# 优化后的内核
@triton.jit
def _vector_add_kernel_optimized(
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_CORES = tl.num_programs(0)
    
    # 跨步分配，充分利用多核
    for block_idx in range(pid, tl.cdiv(n_elements, BLOCK_SIZE), NUM_CORES):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # 向量化加载
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x + y, mask=mask)

# 使用更大的BLOCK_SIZE和更多核心
BLOCK_SIZE = 8192  # 增大块大小
NUM_CORES = 32     # 使用全部AI Core
grid = (NUM_CORES,)
```

### 5.2 Softmax优化

**当前问题**：
- 多遍扫描导致内存带宽浪费
- 未充分利用Vector单元

**优化策略**：
1. **单遍融合内核**：对于小维度，单块处理整行
2. **Online Softmax**：使用在线算法减少内存访问
3. **向量化exp计算**：利用Vector单元加速

**具体实现**：
```python
@triton.jit
def _softmax_kernel_online(
    output_ptr, input_ptr, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Online Softmax - 单遍计算"""
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # Online算法：一次遍历同时计算max和sum
    m_i = float('-inf')  # 当前最大值
    l_i = 0.0            # 当前归一化因子
    
    # 分块处理
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('-inf'))
        
        # Online update
        m_new = tl.maximum(m_i, tl.max(x, 0))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), 0)
        m_i = m_new
    
    # 第二遍：计算并存储结果
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('-inf'))
        result = tl.exp(x - m_i) / l_i
        tl.store(output_ptr + row_start + cols, result, mask=mask)
```

### 5.3 LayerNorm优化

**当前问题**：
- 两遍扫描（统计量 + 归一化）
- 未充分利用Vector单元

**优化策略**：
1. **Welford算法优化**：单遍计算均值和方差
2. **向量化归一化**：利用Vector单元
3. **融合仿射变换**：减少内核启动

**具体实现**：
```python
@triton.jit
def _layer_norm_kernel_fused(
    output_ptr, input_ptr, weight_ptr, bias_ptr,
    n_cols, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """融合LayerNorm - 单块处理整行"""
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    # 加载数据
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Welford算法：单遍计算均值和方差
    mean = tl.sum(x, 0) / n_cols
    var = tl.sum((x - mean) * (x - mean), 0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # 融合归一化和仿射变换
    output = w * (x - mean) * rstd + b
    tl.store(output_ptr + row_start + cols, output, mask=mask)
```

### 5.4 FlashAttention优化

**当前问题**：
- 复杂的分块计算
- 大量中间结果存储
- 未充分利用Cube单元

**优化策略**：
1. **优化分块大小**：避免UB/CBUF溢出
2. **减少中间存储**：优化累加器使用
3. **利用tl.dot**：让编译器生成Cube指令

**具体实现**：
```python
@triton.jit
def _flash_attn_kernel_optimized(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    seq_len, head_dim, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """优化的Flash Attention内核"""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Q块起始位置
    q_start = pid_m * BLOCK_M
    q_rows = q_start + tl.arange(0, BLOCK_M)
    q_mask = q_rows < seq_len
    
    # 加载Q块 - 使用向量化加载
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    q_ptrs = q_ptr + q_offset + q_rows[:, None] * stride_qs + tl.arange(0, BLOCK_D)[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    
    # 初始化累加器
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    
    # 遍历K、V块
    for k_start in range(0, seq_len, BLOCK_N):
        k_rows = k_start + tl.arange(0, BLOCK_N)
        k_mask = k_rows < seq_len
        
        # 加载K块
        k_offset = pid_b * stride_kb + pid_h * stride_kh
        k_ptrs = k_ptr + k_offset + k_rows[:, None] * stride_ks + tl.arange(0, BLOCK_D)[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)
        
        # 使用tl.dot进行矩阵乘法 - 编译器会生成Cube指令
        qk = tl.dot(q, tl.trans(k)) * scale
        
        # Online Softmax更新
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        
        # 加载V块
        v_offset = pid_b * stride_vb + pid_h * stride_vh
        v_ptrs = v_ptr + v_offset + k_rows[:, None] * stride_vs + tl.arange(0, BLOCK_D)[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)
        
        # 更新累加器
        acc = acc * alpha[:, None] + tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
    
    # 归一化并存储
    acc = acc / l_i[:, None]
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    o_ptrs = o_ptr + o_offset + q_rows[:, None] * stride_os + tl.arange(0, BLOCK_D)[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=q_mask[:, None])
```

### 5.5 Matmul优化

**当前问题**：
- 分块参数固定
- 未充分利用Cube单元
- L2缓存利用率低

**优化策略**：
1. **Super-Grouping**：优化L2缓存访问
2. **Auto-tuning**：自动搜索最优分块参数
3. **使用tl.dot**：生成Cube指令

**具体实现**：
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel_optimized(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """优化的矩阵乘法内核 - 使用Super-Grouping"""
    # Super-Grouping: 重新排列program_id以优化L2缓存
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # ... 其余实现
```

## 六、实施计划

### 阶段1：基础优化（优先级最高）
1. **增大BLOCK_SIZE**：减少内核启动开销
2. **优化grid配置**：充分利用32个AI Core
3. **数据对齐**：确保内存访问效率

### 阶段2：算法优化
1. **Online算法**：Softmax/LayerNorm使用在线算法
2. **融合内核**：减少内核启动次数
3. **向量化计算**：充分利用Vector单元

### 阶段3：高级优化
1. **Auto-tuning**：自动搜索最优配置
2. **Super-Grouping**：优化L2缓存
3. **Cube单元利用**：Matmul/Attention使用tl.dot

### 阶段4：迭代调优
1. 运行性能测试
2. 分析瓶颈
3. 调整参数
4. 重复直到达标

## 七、预期效果

| 算子 | 当前 | 优化后预期 | 关键优化点 |
|------|------|-----------|-----------|
| VectorAdd | 0.40x | >0.95x | 大BLOCK_SIZE + 多核并行 |
| Softmax | 0.25x | >0.95x | Online算法 + 向量化 |
| LayerNorm | 0.32x | >0.95x | Welford + 融合内核 |
| FlashAttention | 0.13x | >0.95x | 分块优化 + tl.dot |

## 八、风险与应对

### 风险1：Triton-Ascend编译器限制
- **风险**：某些优化可能不被编译器支持
- **应对**：查阅Triton-Ascend文档，使用支持的特性

### 风险2：硬件特性差异
- **风险**：Ascend 910B4与NVIDIA GPU架构差异大
- **应对**：针对Ascend特性调整优化策略

### 风险3：性能调优周期长
- **风险**：Auto-tuning需要大量测试
- **应对**：先手动调优找到大致范围，再精细调整

## 九、参考资料

1. Triton-Ascend官方文档
2. 昇腾AI处理器架构指南
3. CANN开发文档
4. Flash Attention论文
5. Online Softmax算法