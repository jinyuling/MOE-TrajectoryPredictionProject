#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目运行脚本 - 提供完整的项目运行指南
"""

import os
import sys
import subprocess
import platform

def print_header(title):
    """打印标题"""
    print("\n" + "="*50)
    print(f"🚀 {title}")
    print("="*50)

def detect_os():
    """检测操作系统"""
    system = platform.system()
    print(f"💻 操作系统: {system}")
    return system

def check_python():
    """检查Python环境"""
    print_header("检查Python环境")
    
    python_version = sys.version
    print(f"🐍 Python版本: {python_version}")
    
    if sys.version_info < (3, 7):
        print("❌ Python版本过低，请升级到3.7或更高版本")
        return False
    else:
        print("✅ Python版本满足要求")
        return True

def install_with_conda():
    """使用conda安装依赖（推荐在Windows上使用）"""
    print_header("使用Conda安装依赖（推荐）")
    
    commands = [
        "conda create -n unitraj python=3.9 -y",
        "conda activate unitraj",
        "conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y",
        "pip install pytorch-lightning numpy pandas wandb hydra-core omegaconf"
    ]
    
    print("请依次运行以下命令:")
    for cmd in commands:
        print(f"   {cmd}")
    
    print("\n💡 提示: 如果没有安装conda，请先安装Anaconda或Miniconda")

def install_with_pip():
    """使用pip安装依赖"""
    print_header("使用Pip安装依赖")
    
    commands = [
        "pip install torch torchvision torchaudio",
        "pip install -r requirements.txt"
    ]
    
    print("请依次运行以下命令:")
    for cmd in commands:
        print(f"   {cmd}")

def install_dependencies():
    """安装项目依赖"""
    print_header("安装项目依赖")
    
    system = detect_os()
    
    if system == "Windows":
        print("💡 在Windows上推荐使用conda来避免DLL加载问题")
        install_with_conda()
    else:
        print("💡 在Linux/Mac上可以使用pip安装")
        install_with_pip()

def run_tests():
    """运行测试"""
    print_header("运行测试")
    
    test_scripts = [
        ("环境检查", "python check_env.py"),
        ("代码结构测试", "python test_structure.py"),
        ("简化MOE演示", "python simple_moe.py")
    ]
    
    for name, cmd in test_scripts:
        print(f"\n🧪 {name}:")
        print(f"   命令: {cmd}")
        try:
            # 这里只是打印命令，实际运行需要用户手动执行
            print(f"   状态: 准备就绪")
        except Exception as e:
            print(f"   状态: 错误 - {e}")

def run_training():
    """运行训练"""
    print_header("运行训练")
    
    print("🔧 训练命令:")
    print("   python train.py method=MOE")
    
    print("\n📋 训练配置:")
    print("   - 确保数据集已准备")
    print("   - 确保预训练专家模型权重已下载")
    print("   - 检查配置文件是否正确")

def run_debug():
    """运行调试"""
    print_header("运行调试")
    
    debug_scripts = [
        ("路由选择器调试", "python debug_router.py"),
        ("MOE模型测试", "python test_moe.py")
    ]
    
    for name, cmd in debug_scripts:
        print(f"\n🐛 {name}:")
        print(f"   命令: {cmd}")

def show_project_structure():
    """显示项目结构"""
    print_header("项目结构")
    
    structure = """
    MOE项目/
    ├── base_model.py          # 基础模型类
    ├── moe.py                # MOE模型实现
    ├── train.py              # 训练脚本
    ├── requirements.txt      # 依赖列表
    ├── README.md            # 项目说明
    ├── IMPROVEMENTS.md      # 改进说明
    ├── check_env.py         # 环境检查
    ├── test_structure.py    # 结构测试
    ├── test_moe.py          # MOE测试
    ├── debug_router.py      # 路由调试
    ├── simple_moe.py        # 简化MOE演示
    └── run_project.py       # 运行指南
    """
    
    print(structure)

def show_troubleshooting():
    """显示故障排除指南"""
    print_header("故障排除指南")
    
    troubleshooting = """
常见问题及解决方案:

1. ❌ DLL加载错误 (Windows)
   💡 解决方案: 使用conda安装PyTorch
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

2. ❌ ImportError: No module named 'xxx'
   💡 解决方案: 安装缺失的依赖
   pip install -r requirements.txt

3. ❌ CUDA out of memory
   💡 解决方案: 减小批次大小
   在配置文件中设置 train_batch_size 和 eval_batch_size 为更小的值

4. ❌ Loss不下降
   💡 解决方案: 
   - 检查学习率设置
   - 验证数据加载是否正确
   - 确认模型参数是否正确更新

5. ❌ 路由选择器不工作
   💡 解决方案:
   - 检查路由概率是否正确计算
   - 验证专家模型是否正确冻结
   - 确认损失函数组合是否正确
    """
    
    print(troubleshooting)

def main():
    """主函数"""
    print("🎯 MOE项目完整运行指南")
    
    # 显示项目结构
    show_project_structure()
    
    # 检查环境
    if not check_python():
        print("❌ 环境检查失败，请先解决环境问题")
        return
    
    # 安装依赖说明
    install_dependencies()
    
    # 运行测试
    run_tests()
    
    # 运行训练
    run_training()
    
    # 运行调试
    run_debug()
    
    # 故障排除
    show_troubleshooting()
    
    print("\n" + "="*50)
    print("📋 使用建议:")
    print("1. 首先运行 check_env.py 检查环境")
    print("2. 运行 simple_moe.py 验证基本功能")
    print("3. 安装完整依赖后运行完整训练")
    print("4. 遇到问题时参考故障排除指南")
    print("="*50)

if __name__ == "__main__":
    main()