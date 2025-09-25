#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境检查脚本 - 用于检查项目依赖是否正确安装
"""

import sys
import importlib.util

def check_python_version():
    """检查Python版本"""
    print("🔍 检查Python版本...")
    print(f"Python版本: {sys.version}")
    
    if sys.version_info < (3, 7):
        print("❌ Python版本过低，建议使用Python 3.7或更高版本")
        return False
    else:
        print("✅ Python版本满足要求")
        return True

def check_package(package_name, install_name=None):
    """检查包是否已安装"""
    if install_name is None:
        install_name = package_name
    
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is not None:
            print(f"✅ {package_name} 已安装")
            return True
        else:
            print(f"❌ {package_name} 未安装，请运行: pip install {install_name}")
            return False
    except ImportError:
        print(f"❌ {package_name} 未安装，请运行: pip install {install_name}")
        return False

def check_dependencies():
    """检查项目依赖"""
    print("\n🔍 检查项目依赖...")
    
    dependencies = [
        ("torch", "torch"),
        ("pytorch_lightning", "pytorch-lightning"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("wandb", "wandb"),
        ("hydra", "hydra-core"),
        ("omegaconf", "omegaconf")
    ]
    
    all_good = True
    for package_name, install_name in dependencies:
        if not check_package(package_name, install_name):
            all_good = False
    
    return all_good

def check_files():
    """检查必要的文件是否存在"""
    print("\n🔍 检查必要文件...")
    
    required_files = [
        "base_model.py",
        "moe.py",
        "train.py"
    ]
    
    all_good = True
    for file in required_files:
        try:
            with open(file, 'r') as f:
                print(f"✅ {file} 存在")
        except FileNotFoundError:
            print(f"❌ {file} 不存在")
            all_good = False
    
    return all_good

def main():
    print("🚀 MOE项目环境检查")
    print("=" * 40)
    
    python_ok = check_python_version()
    deps_ok = check_dependencies()
    files_ok = check_files()
    
    print("\n" + "=" * 40)
    if python_ok and deps_ok and files_ok:
        print("🎉 环境检查通过！项目应该可以正常运行。")
        print("\n下一步建议:")
        print("1. 如果要训练模型，运行: python train.py method=MOE")
        print("2. 如果要测试路由选择器，运行: python debug_router.py")
        print("3. 如果要检查代码结构，运行: python test_structure.py")
    else:
        print("⚠️ 环境检查未通过，请根据上面的提示安装缺失的依赖。")
        print("\n环境设置建议:")
        print("1. 推荐使用conda创建虚拟环境:")
        print("   conda create -n unitraj python=3.9")
        print("   conda activate unitraj")
        print("2. 安装PyTorch (Windows推荐使用conda避免DLL问题):")
        print("   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
        print("3. 安装其他依赖:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()