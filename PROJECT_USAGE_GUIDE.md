# MOE轨迹预测项目使用指南

## 项目概述

本指南将帮助您了解如何使用MOE（Mixture of Experts）轨迹预测项目，包括环境设置、代码运行、训练模型和故障排除。

## 1. 环境设置

### 1.1 系统要求
- Windows 10/11 或 Linux/macOS
- Python 3.7+
- 至少8GB内存（推荐16GB以上）
- GPU支持（推荐，但CPU也可运行）

### 1.2 依赖安装

#### 推荐方式：使用Conda（特别是Windows用户）
```bash
# 创建虚拟环境
conda create -n unitraj python=3.9
conda activate unitraj

# 安装PyTorch（避免DLL问题）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install pytorch-lightning numpy pandas wandb hydra-core omegaconf
```

#### 备选方式：使用Pip
```bash
# 安装PyTorch
pip install torch torchvision torchaudio

# 安装其他依赖
pip install -r requirements.txt
```

### 1.3 环境验证
```bash
# 检查环境配置
python check_env.py

# 自动安装缺失的依赖（可选）
python check_env.py --install
```

## 2. 代码结构说明

### 2.1 核心文件
- `base_model.py`：基础模型类，包含负载平衡损失计算
- `moe.py`：MOE模型实现，包含路由选择器和专家模型
- `train.py`：训练脚本，负责模型训练流程

### 2.2 调试和测试工具
- `simple_moe.py`：简化版MOE模型，无外部依赖
- `simple_train.py`：简化版训练演示
- `debug_router.py`：路由选择器调试工具
- `test_moe.py`：MOE模型测试

### 2.3 环境和配置工具
- `check_env.py`：环境检查和依赖安装
- `setup_windows.py`：Windows特殊设置
- `enable_long_paths.bat`：启用Windows长路径支持

## 3. 运行项目

### 3.1 简化版演示
```bash
# 运行简化版MOE模型演示
python simple_moe.py

# 运行简化版训练演示
python simple_train.py
```

### 3.2 完整训练
```bash
# 运行完整训练
python train.py method=MOE
```

### 3.3 路由选择器调试
```bash
# 调试路由选择器
python debug_router.py
```

## 4. 模型训练配置

### 4.1 配置文件
项目使用Hydra进行配置管理，主要配置文件位于`configs/`目录下。

### 4.2 训练参数调整
在`train.py`中可以调整以下关键参数：
- 学习率
- 批次大小
- 训练轮数
- 优化器设置

### 4.3 专家模型集成
MOE模型支持集成多个专家模型：
- AutoBot
- Wayformer
- 其他轨迹预测模型

## 5. 数据准备

### 5.1 数据格式
项目支持多种轨迹预测数据集格式：
- Waymo Open Motion Dataset
- nuScenes
- Argoverse 2

### 5.2 数据预处理
数据预处理脚本位于`datasets/`目录下，包含：
- 数据加载器
- 数据增强
- 数据标准化

## 6. 模型评估

### 6.1 评估指标
项目支持多种轨迹预测评估指标：
- minADE（最小平均位移误差）
- minFDE（最小最终位移误差）
- Miss Rate（失误率）

### 6.2 可视化
项目集成了WandB用于训练过程可视化和结果展示。

## 7. 故障排除

### 7.1 Windows常见问题

#### 长路径问题
```bash
# 启用Windows长路径支持
python setup_windows.py
# 或运行 enable_long_paths.bat（以管理员身份）
```

#### DLL加载错误
推荐使用conda安装PyTorch：
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 7.2 依赖安装问题

#### 自动安装依赖
```bash
python check_env.py --install
```

#### CPU版本PyTorch（备选方案）
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 7.3 训练问题

#### 损失不下降
- 检查学习率设置
- 验证数据加载是否正确
- 确认模型参数是否正确更新

#### 内存不足
- 减小批次大小
- 使用CPU训练（较慢但可行）
- 优化数据加载器

## 8. 进一步优化

### 8.1 路由选择器优化
- 调整注意力机制参数
- 改进特征提取器
- 优化路由决策网络

### 8.2 专家模型集成
- 添加更多专家模型
- 调整专家模型权重
- 实现动态专家选择

### 8.3 负载平衡机制
- 调整负载平衡损失权重
- 实现更复杂的负载平衡策略
- 监控专家模型使用情况

## 9. 开发和贡献

### 9.1 Git工作流程
```bash
# 创建功能分支
git checkout -b feature/新功能名称

# 提交更改
git add .
git commit -m "描述您的更改"

# 推送到GitHub
git push origin feature/新功能名称
```

### 9.2 代码规范
- 遵循PEP 8 Python代码规范
- 添加适当的注释和文档
- 编写单元测试

## 10. 学习资源

### 10.1 文档
- `IMPROVEMENTS.md`：技术改进详细说明
- `PROJECT_SUMMARY.md`：项目完整总结
- `run_project.py`：完整的运行指南

### 10.2 教程
- `git_tutorial.py`：Git版本控制教程
- `setup_env.py`：环境设置说明

## 11. 项目维护

### 11.1 定期更新
- 更新依赖包到最新稳定版本
- 修复已知问题
- 添加新功能

### 11.2 性能监控
- 监控训练性能
- 优化模型推理速度
- 减少内存占用

通过遵循本指南，您应该能够成功运行和使用MOE轨迹预测项目。如果遇到任何问题，请参考故障排除部分或提交GitHub Issue。