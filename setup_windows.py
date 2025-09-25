#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows环境设置脚本 - 解决长路径问题和依赖安装
"""

import os
import sys
import subprocess
import platform
import winreg

def check_windows_version():
    """检查Windows版本"""
    print("🔍 检查Windows版本...")
    system = platform.system()
    release = platform.release()
    
    if system == "Windows":
        print(f"✅ 系统: Windows {release}")
        return True
    else:
        print(f"❌ 当前系统不是Windows: {system}")
        return False

def check_long_path_enabled():
    """检查Windows长路径支持是否已启用"""
    print("\n🔍 检查Windows长路径支持...")
    
    try:
        # 打开注册表项
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem"
        )
        
        # 读取长路径策略值
        value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        winreg.CloseKey(key)
        
        if value == 1:
            print("✅ Windows长路径支持已启用")
            return True
        else:
            print("❌ Windows长路径支持未启用")
            return False
    except Exception as e:
        print(f"⚠️ 无法检查长路径设置: {e}")
        return False

def enable_long_path():
    """启用Windows长路径支持"""
    print("\n🔧 启用Windows长路径支持...")
    
    try:
        # 使用PowerShell命令启用长路径支持
        cmd = [
            "powershell", 
            "-Command", 
            "Set-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' -Name 'LongPathsEnabled' -Value 1"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            print("✅ Windows长路径支持已启用")
            print("💡 请重启计算机以使更改生效")
            return True
        else:
            print(f"❌ 启用长路径支持失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 启用长路径支持时出错: {e}")
        return False

def check_pip_version():
    """检查pip版本"""
    print("\n🔍 检查pip版本...")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ pip版本: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ 无法获取pip版本: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 检查pip版本时出错: {e}")
        return False

def upgrade_pip():
    """升级pip"""
    print("\n🔧 升级pip...")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ pip升级成功")
            return True
        else:
            print(f"❌ pip升级失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 升级pip时出错: {e}")
        return False

def install_with_conda_recommended():
    """推荐使用conda安装的说明"""
    print("\n💡 推荐使用conda安装PyTorch相关依赖")
    print("\n步骤:")
    print("1. 下载并安装Anaconda或Miniconda:")
    print("   访问 https://www.anaconda.com/products/distribution 或 https://docs.conda.io/en/latest/miniconda.html")
    print("2. 创建虚拟环境:")
    print("   conda create -n unitraj python=3.9")
    print("   conda activate unitraj")
    print("3. 安装PyTorch (避免DLL问题):")
    print("   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    print("4. 安装其他依赖:")
    print("   pip install pytorch-lightning pandas wandb hydra-core omegaconf")

def install_alternative_torch():
    """安装CPU版本的PyTorch作为替代方案"""
    print("\n🔧 尝试安装CPU版本的PyTorch...")
    
    try:
        print("🔄 安装CPU版本的PyTorch...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ CPU版本PyTorch安装成功")
            return True
        else:
            print(f"❌ CPU版本PyTorch安装失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 安装CPU版本PyTorch时出错: {e}")
        return False

def main():
    """主函数"""
    print("🔧 Windows MOE项目环境设置")
    print("=" * 40)
    
    # 检查系统
    if not check_windows_version():
        print("❌ 不支持的操作系统")
        return
    
    # 检查长路径支持
    long_path_enabled = check_long_path_enabled()
    
    # 检查pip
    check_pip_version()
    
    print("\n" + "=" * 40)
    print("🛠 解决方案建议:")
    
    if not long_path_enabled:
        print("\n1. 启用Windows长路径支持:")
        print("   运行以下命令作为管理员:")
        print("   Set-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' -Name 'LongPathsEnabled' -Value 1")
        print("   或者运行此脚本中的启用功能")
        
        # 询问是否要启用
        choice = input("\n是否要尝试启用长路径支持？(y/n): ").lower()
        if choice == 'y':
            enable_long_path()
    
    print("\n2. 升级pip:")
    choice = input("是否要升级pip？(y/n): ").lower()
    if choice == 'y':
        upgrade_pip()
    
    print("\n3. 替代安装方案:")
    print("   如果仍然遇到问题，可以尝试安装CPU版本的PyTorch")
    choice = input("是否要尝试安装CPU版本的PyTorch？(y/n): ").lower()
    if choice == 'y':
        install_alternative_torch()
    
    print("\n4. 推荐方案:")
    install_with_conda_recommended()
    
    print("\n" + "=" * 40)
    print("📋 验证步骤:")
    print("1. 运行 python check_env.py 检查环境")
    print("2. 运行 python simple_moe.py 验证基本功能")
    print("3. 运行 python simple_train.py 验证训练流程")
    
    print("\n💡 提示:")
    print("即使某些依赖安装失败，您仍然可以使用简化版的MOE模型进行学习和测试。")

if __name__ == "__main__":
    # 检查是否以管理员权限运行
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        if not is_admin:
            print("⚠️  某些操作可能需要管理员权限")
    except:
        pass
    
    main()