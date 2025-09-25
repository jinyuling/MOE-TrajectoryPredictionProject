"""
模型构建模块
"""

def build_model(cfg):
    """
    构建模型
    
    Args:
        cfg: 配置对象
        
    Returns:
        构建的模型
    """
    # 导入放在函数内部，避免循环导入
    from moe import MOE, TrajAttentionRouter
    
    # 简化处理，直接创建MOE模型
    model = MOE(cfg.method)
    return model

def build_dataset(cfg, val=False):
    """
    构建数据集
    
    Args:
        cfg: 配置对象
        val: 是否为验证集
        
    Returns:
        构建的数据集
    """
    # 创建模拟数据集用于演示
    from dummy_dataset import DummyDataset
    return DummyDataset(cfg, val)