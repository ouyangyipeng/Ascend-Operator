# Ascend Operator Library
# 基于Triton-Ascend构建昇腾亲和算子

# 工具函数
from .utils import (
    has_npu_driver,
    get_device,
    get_device_properties,
    get_num_cores,
)

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
    """延迟导入算子函数
    
    由于模块名和函数名冲突（如 operators.vector_add 既是模块也是函数），
    我们需要直接从模块字典中获取函数对象并缓存到当前模块。
    """
    import sys
    
    # 算子名称到模块名的映射
    operator_map = {
        "vector_add": ("vector_add", "vector_add"),
        "vector_add_autotuned": ("vector_add", "vector_add_autotuned"),
        "matmul": ("matmul", "matmul"),
        "matmul_reference": ("matmul", "matmul_reference"),
        "softmax": ("softmax", "softmax"),
        "softmax_reference": ("softmax", "softmax_reference"),
        "layer_norm": ("layer_norm", "layer_norm"),
        "layer_norm_reference": ("layer_norm", "layer_norm_reference"),
        "rms_norm": ("rms_norm", "rms_norm"),
        "rms_norm_reference": ("rms_norm", "rms_norm_reference"),
        "flash_attention": ("flash_attention", "flash_attention"),
        "flash_attention_reference": ("flash_attention", "flash_attention_reference"),
        "reduce_sum": ("reduction", "reduce_sum"),
        "reduce_max": ("reduction", "reduce_max"),
        "reduce_min": ("reduction", "reduce_min"),
    }
    
    if name not in operator_map:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    
    module_name, func_name = operator_map[name]
    full_module_name = f"operators.{module_name}"
    
    # 导入模块
    if full_module_name not in sys.modules:
        import importlib
        importlib.import_module(f".{module_name}", "operators")
    
    # 从模块中获取函数
    mod = sys.modules[full_module_name]
    func = getattr(mod, func_name)
    
    # 缓存到当前模块的 __dict__ 中，避免后续 __getattr__ 调用
    globals()[name] = func
    
    return func