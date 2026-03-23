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
        import torch_npu
        return torch.npu.is_available()
    except (ImportError, RuntimeError):
        return False


def get_device():
    """
    获取可用设备
    
    Returns:
        str: 'npu:0' 如果有NPU，否则返回 'cpu'
    """
    if has_npu_driver():
        return 'npu:0'
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
            props = {
                'name': torch.npu.get_device_name(0),
                'capability': torch.npu.get_device_capability(0),
                'memory': torch.npu.get_device_properties(0).total_memory,
                'multi_processor_count': torch.npu.get_device_properties(0).multi_processor_count,
            }
            return props
        except (ImportError, RuntimeError, AttributeError):
            pass
    return {'name': 'cpu', 'capability': None, 'memory': 0, 'multi_processor_count': 0}


# 全局变量缓存物理核数
_num_cores = None


def _init_globals():
    """初始化全局变量"""
    global _num_cores
    if _num_cores is not None:
        return
    
    try:
        # 尝试从系统获取物理核数
        import os
        # 获取物理CPU核心数
        _num_cores = os.cpu_count() or 1
        
        # 尝试获取物理核数（不包括超线程）
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read()
            # 统计物理处理器数量
            processors = content.count('processor')
            if processors > 0:
                _num_cores = processors
        except:
            pass
    except:
        _num_cores = 1


def get_num_cores():
    """
    获取CPU物理核数
    
    Returns:
        int: CPU物理核数
    """
    global _num_cores
    if _num_cores is None:
        _init_globals()
    return _num_cores


if __name__ == "__main__":
    print(f"Has NPU driver: {has_npu_driver()}")
    print(f"Device: {get_device()}")
    print(f"Device properties: {get_device_properties()}")
    print(f"Number of CPU cores: {get_num_cores()}")