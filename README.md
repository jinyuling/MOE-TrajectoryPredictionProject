# MOE 路由选择器调试和改进

这个项目是针对MOE（Mixture of Experts）架构中的路由选择器进行调试和改进的代码库。

## 问题分析

在原始实现中，路由选择器训练存在以下问题：

1. **损失无法下降**：路由选择器的损失在训练过程中无法有效下降
2. **性能下降**：使用MOE后性能比单独使用专家模型更差
3. **路由不生效**：路由选择器未能有效选择最适合的专家模型

## 改进方案

### 1. 路由选择器架构改进

在 [moe.py](file:///c:/Users/Administrator/Desktop/01/moe.py) 中，我们改进了 [TrajAttentionRouter](file:///c:/Users/Administrator/Desktop/01/moe.py#L24-L132) 类：

- 增强了轨迹特征提取能力
- 改进了路网特征编码
- 优化了注意力机制的使用
- 添加了更好的权重初始化

### 2. 损失函数优化

在 [base_model.py](file:///c:/Users/Administrator/Desktop/01/base_model.py) 中，我们改进了损失计算：

- 修正了负载平衡损失的计算方式
- 添加了更详细的日志记录
- 改进了损失权重的平衡

### 3. 训练策略调整

在 [train.py](file:///c:/Users/Administrator/Desktop/01/train.py) 中，我们保持了专家模型冻结、只训练路由选择器的策略，但添加了更好的日志记录。

## 使用方法

### 1. 运行测试脚本

```bash
python test_moe.py
```

这个脚本会测试路由选择器的基本功能。

### 2. 运行调试脚本

```bash
python debug_router.py
```

这个脚本会进行路由选择器的简化训练，验证其学习能力。

### 3. 完整训练

使用原始的训练脚本：

```bash
python train.py method=MOE
```

## 关键改进点

### 1. 特征提取增强

```python
# 改进的轨迹特征提取器
self.trajectory_encoder = nn.Sequential(
    nn.Linear(self.k_attr, self.d_k),
    nn.ReLU(),
    nn.LayerNorm(self.d_k),
    nn.Linear(self.d_k, self.d_k),
    nn.ReLU(),
    nn.LayerNorm(self.d_k)
)
```

### 2. 负载平衡损失修正

```python
def load_balance_loss(self, routing_probs):
    """计算负载平衡损失，鼓励专家均匀使用"""
    # 计算每个专家的平均使用概率
    expert_mean = routing_probs.mean(dim=0)
    # 计算负载平衡损失
    loss = (expert_mean * routing_probs.sum(dim=0)).sum() / routing_probs.size(0)
    return loss
```

### 3. 更好的日志记录

添加了详细的训练日志记录，包括：
- 路由概率分布
- 各种损失组件
- 训练进度跟踪

## 预期效果

通过这些改进，预期能够：

1. **降低训练损失**：路由选择器的损失应该能够有效下降
2. **提高模型性能**：MOE模型的性能应该至少不低于单独的专家模型
3. **有效路由选择**：路由选择器应该能够根据不同输入选择最适合的专家

## 故障排除

如果仍然遇到问题，请检查：

1. **数据格式**：确保输入数据格式正确
2. **学习率**：尝试调整路由选择器的学习率
3. **损失权重**：调整负载平衡损失的权重
4. **特征提取**：验证特征提取器是否能有效提取轨迹特征

## 进一步改进

如果基础改进仍然无法解决问题，可以考虑：

1. **联合训练**：同时训练路由选择器和专家模型
2. **更复杂的路由机制**：使用更高级的路由算法
3. **专家 specialize**：让专家模型 specialize 于不同的轨迹类型