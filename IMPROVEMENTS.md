# MOE路由选择器改进说明

## 问题分析

在原始实现中，路由选择器训练存在以下问题：

1. **损失无法下降**：路由选择器的损失在训练过程中无法有效下降
2. **性能下降**：使用MOE后性能比单独使用专家模型更差
3. **路由不生效**：路由选择器未能有效选择最适合的专家模型

## 改进方案

### 1. 路由选择器架构改进

在 [moe.py](file:///c:/Users/Administrator/Desktop/01/moe.py) 中，我们对 [TrajAttentionRouter](file:///c:/Users/Administrator/Desktop/01/moe.py#L24-L132) 类进行了以下改进：

#### 特征提取增强
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

# 路网特征提取
self.road_encoder = nn.Sequential(
    nn.Linear(self.map_attr, self.d_k),
    nn.ReLU(),
    nn.LayerNorm(self.d_k)
)
```

#### 注意力机制优化
```python
# 使用Perceiver编码器进行特征融合
self.perceiver_encoder = PerceiverEncoder(192, self.d_k,
                                         num_cross_attention_qk_channels=self.d_k,
                                         num_cross_attention_v_channels=self.d_k,
                                         num_self_attention_qk_channels=self.d_k,
                                         num_self_attention_v_channels=self.d_k)
```

#### 路由决策网络
```python
# 改进的路由决策网络
self.routing_head = nn.Sequential(
    nn.Linear(self.d_k, self.d_k // 2),
    nn.ReLU(),
    nn.LayerNorm(self.d_k // 2),
    nn.Linear(self.d_k // 2, self.num_experts)
)
```

### 2. 损失函数优化

在 [base_model.py](file:///c:/Users/Administrator/Desktop/01/base_model.py) 中，我们改进了损失计算：

#### 负载平衡损失修正
```python
def load_balance_loss(self, routing_probs):
    """计算负载平衡损失，鼓励专家均匀使用"""
    # 计算每个专家的平均使用概率
    expert_mean = routing_probs.mean(dim=0)
    # 计算负载平衡损失
    loss = (expert_mean * routing_probs.sum(dim=0)).sum() / routing_probs.size(0)
    return loss
```

#### 损失组合优化
```python
# 添加负载平衡损失
lb_loss = self.load_balance_loss(routing_probs)
lb_loss_weight = 0.01  # 负载平衡损失权重
total_loss = loss + lb_loss_weight * lb_loss
```

### 3. 训练策略调整

在 [moe.py](file:///c:/Users/Administrator/Desktop/01/moe.py) 中，我们优化了MOE模型的训练策略：

#### 专家模型冻结
```python
def freeze_experts(self):
    """冻结所有专家的参数"""
    for idx, expert in enumerate(self.experts):
        for param in expert.parameters():
            param.requires_grad = False
        print(f"🔒 Expert {idx} parameters frozen")
```

#### 参数优化器配置
```python
def configure_optimizers(self):
    if self.train_router_only:
        # 只优化 Router 参数
        trainable_params = list(self.router.parameters())
        optimizer = optim.Adam(trainable_params, lr=self.config['learning_rate'], eps=0.0001, weight_decay=1e-4)
        print(f"✅ Optimizing {len(trainable_params)} Router parameters only")
    else:
        # 优化所有参数
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], eps=0.0001, weight_decay=1e-4)
        print(f"✅ Optimizing all parameters")
```

### 4. 日志记录增强

在 [base_model.py](file:///c:/Users/Administrator/Desktop/01/base_model.py) 中，我们添加了更详细的日志记录：

#### 路由概率记录
```python
# 记录路由概率分布
routing_mean = routing_probs.mean(dim=0)
for i in range(routing_mean.size(0)):
    self.log(f'train/routing_prob_expert_{i}', routing_mean[i], on_step=False, on_epoch=True)
```

#### 损失组件记录
```python
self.log('train/lb_loss', lb_loss, on_step=False, on_epoch=True)
self.log('train/total_loss', total_loss, on_step=False, on_epoch=True)
```

#### CSV日志保存
```python
# 保存损失到CSV文件
csv_path = './logs/moe_loss.csv'
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
write_header = not os.path.exists(csv_path)

with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(['epoch', 'batch_idx', 'total_loss', 'main_loss', 'lb_loss'])  # 表头
    writer.writerow([self.current_epoch, batch_idx, total_loss.item(), loss.item(), lb_loss.item()])
```

## 预期效果

通过这些改进，预期能够：

1. **降低训练损失**：路由选择器的损失应该能够有效下降
2. **提高模型性能**：MOE模型的性能应该至少不低于单独的专家模型
3. **有效路由选择**：路由选择器应该能够根据不同输入选择最适合的专家
4. **更好的负载平衡**：专家模型的使用应该更加均衡

## 使用方法

### 1. 环境设置

```bash
# 推荐使用conda创建环境
conda create -n unitraj python=3.9
conda activate unitraj

# 安装PyTorch (推荐使用conda以避免Windows上的DLL问题)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 运行测试

```bash
# 测试代码结构
python test_structure.py

# 运行MOE测试（需要完整环境）
python test_moe.py

# 调试路由选择器
python debug_router.py
```

### 3. 完整训练

```bash
# 训练MOE模型
python train.py method=MOE
```

## 故障排除

如果仍然遇到问题，请检查：

1. **数据格式**：确保输入数据格式正确
2. **学习率**：尝试调整路由选择器的学习率（建议从0.001开始）
3. **损失权重**：调整负载平衡损失的权重（建议从0.01开始）
4. **特征提取**：验证特征提取器是否能有效提取轨迹特征
5. **梯度问题**：检查是否存在梯度消失或爆炸问题

## 进一步改进

如果基础改进仍然无法解决问题，可以考虑：

1. **联合训练**：同时训练路由选择器和专家模型
2. **更复杂的路由机制**：使用更高级的路由算法（如Top-K路由、噪声路由等）
3. **专家 specialize**：让专家模型 specialize 于不同的轨迹类型
4. **动态路由**：根据输入动态调整路由策略
5. **路由温度控制**：添加温度参数控制路由的softness