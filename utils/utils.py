"""
工具函数模块
"""

import torch
import random
import numpy as np
import os
import glob

def set_seed(seed):
    """
    设置随机种子
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_latest_checkpoint(pattern):
    """
    查找最新的检查点文件
    
    Args:
        pattern: 文件匹配模式
        
    Returns:
        最新的检查点文件路径，如果没有找到则返回None
    """
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    
    # 按修改时间排序，返回最新的文件
    latest_file = max(files, key=os.path.getmtime)
    return latest_file