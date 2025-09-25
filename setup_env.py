#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境设置说明脚本
"""

import sys
import subprocess
import platform

def check_python_version():
    """检查Python版本"""
    print(f"Python版本: {sys.version}")
    if sys.version_info < (3, 7):
        print("警告: 建议使用Python 3.7或更高版本")
    else:
        print("✅ Python版本满足要求")

def check_system():
    """检查系统信息"""
    system = platform.system()
    print(f"操作系统: {system}")
    print(f"系统架构: {platform.architecture()[0]}")
    
    if system == "Windows":
        print("💡 在Windows上建议使用conda环境来避免DLL问题")

def print_setup_instructions():
    """打印环境设置说明"""
    print("\n" + "="*50)
    print("环境设置说明")
    print("="*50)
    
    print("\n1. 推荐使用conda创建环境:")
    print("   conda create -n unitraj python=3.9")
    print("   conda activate unitraj")
    
    print("\n2. 安装PyTorch (推荐使用conda以避免Windows上的DLL问题):")
    print("   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    print("   或者使用pip:")
    print("   pip install torch torchvision torchaudio")
    
    print("\n3. 安装其他依赖:")
    print("   pip install -r requirements.txt")
    
    print("\n4. 如果遇到CUDA相关问题，可以安装CPU版本:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    
    print("\n5. 运行测试:")
    print("   python test_moe.py")

def main():
    print("MOE项目环境检查")
    print("="*30)
    
    check_python_version()
    check_system()
    print_setup_instructions()
    
    print("\n💡 提示: 请按照上述说明设置环境后再运行测试脚本")

if __name__ == "__main__":
    main()