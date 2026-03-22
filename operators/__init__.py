# Ascend Operator Library
# 基于Triton-Ascend构建昇腾亲和算子

# 工具函数
from .utils import (
    has_npu_driver,
    get_device,
    get_device_properties,
    get_num_cores,
)

# 延迟导入算子，避免在没有NPU驱动时导入失败
__all__ = [
    # 工具函数
    "has_npu_driver",
    "get_device",
    "get_device_properties",
    "get_num_cores",
    # 向量运算
    "vector_add",
    "vector_add_autotuned",
    # 矩阵运算
    "matmul",
    "matmul_reference",
    # 归一化
    "softmax",
    "softmax_reference",
    "layer_norm",
    "layer_norm_reference",
    "rms_norm",
    "rms_norm_reference",
    # 注意力
    "flash_attention",
    "flash_attention_reference",
    # 归约
    "reduce_sum",
    "reduce_max",
    "reduce_min",
]

__version__ = "0.1.0"


def __getattr__(name):
    """延迟导入算子"""
    if name == "vector_add":
        from .vector_add import vector_add
        return vector_add
    elif name == "vector_add_autotuned":
        from .vector_add import vector_add_autotuned
        return vector_add_autotuned
    elif name == "matmul":
        from .matmul import matmul
        return matmul
    elif name == "matmul_reference":
        from .matmul import matmul_reference
        return matmul_reference
    elif name == "softmax":
        from .softmax import softmax
        return softmax
    elif name == "softmax_reference":
        from .softmax import softmax_reference
        return softmax_reference
    elif name == "layer_norm":
        from .layer_norm import layer_norm
        return layer_norm
    elif name == "layer_norm_reference":
        from .layer_norm import layer_norm_reference
        return layer_norm_reference
    elif name == "rms_norm":
        from .rms_norm import rms_norm
        return rms_norm
    elif name == "rms_norm_reference":
        from .rms_norm import rms_norm_reference
        return rms_norm_reference
    elif name == "flash_attention":
        from .flash_attention import flash_attention
        return flash_attention
    elif name == "flash_attention_reference":
        from .flash_attention import flash_attention_reference
        return flash_attention_reference
    elif name == "reduce_sum":
        from .reduction import reduce_sum
        return reduce_sum
    elif name == "reduce_max":
        from .reduction import reduce_max
        return reduce_max
    elif name == "reduce_min":
        from .reduction import reduce_min
        return reduce_min
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")