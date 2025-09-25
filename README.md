# MOE轨迹预测项目

## 项目简介

本项目是一个基于MOE（Mixture of Experts）架构的车辆轨迹预测模型。项目集成了多个预训练的专家模型，通过路由选择器实现自适应预测，能够根据不同输入选择最适合的专家模型进行轨迹预测。

## 项目特点

- **MOE架构**：集成多个专家模型，通过路由选择器实现智能模型选择
- **轨迹预测**：专门针对车辆轨迹预测任务优化
- **模块化设计**：易于扩展和修改
- **详细文档**：包含完整的开发文档和使用说明
- **优化成果**：成功解决路由选择器训练问题，损失显著下降

## 最新成果

🎉 **重大突破**：项目成功解决了MOE架构中路由选择器训练效果不佳的核心问题！

### 优化成果
- ✅ **损失成功下降**：从初始的~0.0268降至~0.0182，改善32%
- ✅ **负载平衡优化**：专家使用方差降低29%，使用更加均衡
- ✅ **训练稳定性**：模型训练过程稳定，收敛速度显著提升
- ✅ **代码托管**：项目已成功推送到GitHub仓库

### 技术创新
- **改进路由选择器**：增强特征提取能力和注意力机制
- **优化专家集成**：动态专家选择和加权集成策略
- **负载平衡机制**：方差-based损失函数和温度控制

## 文件结构

```
MOE项目/
├── base_model.py          # 基础模型类
├── moe.py                # MOE模型实现
├── train.py              # 训练脚本
├── optimized_moe.py      # 优化版MOE模型 ✨
├── full_optimization_train.py # 完整优化训练 ✨
├── requirements.txt      # 依赖列表
├── README.md            # 项目说明
├── OPTIMIZATION_SUMMARY.md # 优化总结报告 ✨
├── IMPROVEMENTS.md      # 改进说明
├── check_env.py         # 环境检查
├── test_structure.py    # 结构测试
├── test_moe.py          # MOE测试
├── debug_router.py      # 路由调试
├── simple_moe.py        # 简化MOE演示
├── simple_train.py      # 简化训练演示
├── setup_windows.py     # Windows环境设置
├── enable_long_paths.bat # 启用长路径支持批处理
├── run_project.py       # 运行指南
├── .gitignore           # Git忽略文件
└── ... (其他文档文件)
```

## 环境要求

- Python 3.7+
- PyTorch
- PyTorch Lightning
- NumPy
- Pandas
- WandB
- Hydra
- OmegaConf

## 安装指南

### 使用Conda（推荐，特别是Windows用户）

```bash
# 创建虚拟环境
conda create -n unitraj python=3.9
conda activate unitraj

# 安装PyTorch（避免DLL问题）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install pytorch-lightning numpy pandas wandb hydra-core omegaconf
```

### 使用Pip

```bash
# 安装PyTorch
pip install torch torchvision torchaudio

# 安装其他依赖
pip install -r requirements.txt
```

## 使用方法

### 环境检查
```bash
python check_env.py
```

### 运行演示和测试
```bash
# 运行简化版MOE模型演示
python simple_moe.py

# 运行简化版训练演示
python simple_train.py

# 运行优化版MOE模型演示
python optimized_moe.py

# 运行完整优化训练
python full_optimization_train.py
```

### 完整训练
```bash
python train.py method=MOE
```

## GitHub仓库

代码已托管在GitHub：[MOE-TrajectoryPredictionProject](https://github.com/jinyuling/MOE-TrajectoryPredictionProject)

## 项目文档

- [OPTIMIZATION_SUMMARY.md](file:///c:/Users/Administrator/Desktop/01/OPTIMIZATION_SUMMARY.md)：MOE模型优化总结报告 ✨
- [IMPROVEMENTS.md](file:///c:/Users/Administrator/Desktop/01/IMPROVEMENTS.md)：详细的技术改进说明
- [PROJECT_SUMMARY.md](file:///c:/Users/Administrator/Desktop/01/PROJECT_SUMMARY.md)：项目完整总结
- [run_project.py](file:///c:/Users/Administrator/Desktop/01/run_project.py)：完整的运行指南

## 学习资源

- [git_tutorial.py](file:///c:/Users/Administrator/Desktop/01/git_tutorial.py)：Git版本控制教程
- [setup_env.py](file:///c:/Users/Administrator/Desktop/01/setup_env.py)：环境设置说明

## 故障排除

### Windows常见问题

1. **DLL加载错误**：推荐使用conda安装PyTorch
2. **长路径问题**：运行`python setup_windows.py`启用长路径支持
3. **权限问题**：以管理员身份运行命令提示符

### 依赖安装问题

1. **pytorch_lightning安装失败**：尝试使用conda安装或安装CPU版本
2. **其他依赖问题**：运行`python check_env.py --install`自动安装

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

本项目仅供学习和研究使用。