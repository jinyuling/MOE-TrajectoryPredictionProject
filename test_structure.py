#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试代码结构的简化脚本（不依赖PyTorch）
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入"""
    print("Testing imports...")
    
    try:
        # 测试基础模型导入
        from base_model import BaseModel
        print("✅ BaseModel import successful")
    except Exception as e:
        print(f"❌ BaseModel import failed: {e}")
    
    try:
        # 测试MOE模型导入
        from moe import MOE, TrajAttentionRouter
        print("✅ MOE import successful")
    except Exception as e:
        print(f"❌ MOE import failed: {e}")
    
    try:
        # 测试训练脚本导入
        import train
        print("✅ Train script import successful")
    except Exception as e:
        print(f"❌ Train script import failed: {e}")

def test_file_structure():
    """测试文件结构"""
    print("\nTesting file structure...")
    
    required_files = [
        'base_model.py',
        'moe.py',
        'train.py',
        'test_moe.py',
        'debug_router.py',
        'requirements.txt',
        'README.md'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")

def test_config_structure():
    """测试配置结构"""
    print("\nTesting config structure...")
    
    # 模拟配置
    config = {
        'hidden_size': 256,
        'past_len': 20,
        'num_map_feature': 29,
        'num_agent_feature': 39,
        'num_queries_enc': 192,
        'num_queries_dec': 12,
        'max_num_roads': 32,
        'max_num_agents': 32,
        'num_experts': 2,
        'learning_rate': 0.001,
        'learning_rate_sched': [10, 20, 30],
        'train_router_only': True,
        'method': {
            'model_name': 'MOE'
        }
    }
    
    required_keys = [
        'hidden_size',
        'past_len',
        'num_map_feature',
        'num_agent_feature',
        'num_experts',
        'learning_rate'
    ]
    
    for key in required_keys:
        if key in config:
            print(f"✅ Config key '{key}' present")
        else:
            print(f"❌ Config key '{key}' missing")

def main():
    print("Testing code structure...")
    print("="*40)
    
    test_imports()
    test_file_structure()
    test_config_structure()
    
    print("\n" + "="*40)
    print("🎉 Structure test completed!")

if __name__ == "__main__":
    main()