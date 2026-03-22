"""
通用工具函数
用于检测环境和提供通用功能
"""

import torch


def has_npu_driver():
    """
    检查是否有可用的NPU驱动
    
    Returns:
        bool: 如果有可用的NPU驱动返回True，否则返回False
    """
    try:
        import triton.runtime.driver as driver
        # 检查是否有活跃的驱动
        actives = driver.driver._init_drivers()
        if len(actives) == 0:
            return False
        return True
    except:
        return False


def get_device():
    """
    获取可用设备
    
    Returns:
        str: 'npu:0' 如果有NPU，否则返回 'cpu'
    """
    if has_npu_driver():
        try:
            import torch_npu
            if torch.npu.is_available():
                return 'npu:0'
        except ImportError:
            pass
    return 'cpu'


def get_device_properties():
    """
    获取设备属性
    
    Returns:
        dict: 设备属性字典
    """
    if has_npu_driver():
        try:
            import torch_npu
            import triton.runtime.driver as driver
            device = torch_npu.npu.current_device()
            properties = driver.active.utils.get_device_properties(device)
            return properties
        except:
            pass
    
    # 返回默认属性
    return {
        "num_vectorcore": 1,
        "num_aicore": 1,
        "max_shared_mem": 192 * 1024,  # 192KB
    }


# 全局变量，避免重复检测
_HAS_NPU = None
_DEVICE = None
_DEVICE_PROPERTIES = None


def _init_globals():
    """初始化全局变量"""
    global _HAS_NPU, _DEVICE, _DEVICE_PROPERTIES
    if _HAS_NPU is None:
        _HAS_NPU = has_npu_driver()
    if _DEVICE is None:
        _DEVICE = get_device()
    if _DEVICE_PROPERTIES is None:
        _DEVICE_PROPERTIES = get_device_properties()


def get_num_cores():
    """
    获取物理核数
    
    Returns:
        tuple: (vector_core数量, cube_core数量)
    """
    _init_globals()
    props = _DEVICE_PROPERTIES
    return props.get("num_vectorcore", 1), props.get("num_aicore", 1)


if __name__ == "__main__":
    _init_globals()
    print(f"Has NPU driver: {_HAS_NPU}")
    print(f"Device: {_DEVICE}")
    print(f"Device properties: {_DEVICE_PROPERTIES}")
    print(f"Number of cores: {get_num_cores()}")