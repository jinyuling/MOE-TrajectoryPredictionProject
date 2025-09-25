#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试路由选择器的简化脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import csv
import time

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

def create_dummy_batch(batch_size=4):
    """创建模拟批次数据"""
    input_dict = {
        'obj_trajs': torch.randn(batch_size, 32, 20, 39),
        'obj_trajs_mask': torch.ones(batch_size, 32, 20),
        'map_polylines': torch.randn(batch_size, 32, 20, 29),
        'map_polylines_mask': torch.ones(batch_size, 32, 20),
        'track_index_to_predict': torch.randint(0, 32, (batch_size,)),
        'center_gt_trajs': torch.randn(batch_size, 60, 4),
        'center_gt_trajs_mask': torch.ones(batch_size, 60),
        'center_gt_final_valid_idx': torch.randint(0, 60, (batch_size,)),
        'scenario_id': ['scenario_{}'.format(i) for i in range(batch_size)],
        'center_objects_world': torch.randn(batch_size, 7),
        'center_objects_id': torch.randint(0, 1000, (batch_size,)),
        'center_objects_type': torch.randint(1, 4, (batch_size,)),
        'map_center': torch.randn(batch_size, 2),
        'center_gt_trajs_src': torch.randn(batch_size, 60, 4),
        'dataset_name': ['test'] * batch_size,
        'kalman_difficulty': torch.randint(0, 90, (batch_size, 3)),
        'trajectory_type': torch.randint(0, 8, (batch_size,)),
    }
    
    batch = {
        'input_dict': input_dict,
        'batch_size': batch_size
    }
    
    return batch

class SimpleExpert(nn.Module):
    """简化的专家模型，用于调试"""
    def __init__(self, input_size=39, hidden_size=256, output_size=5):
        super(SimpleExpert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, batch):
        # 简化的前向传播
        input_dict = batch['input_dict']
        obj_trajs = input_dict['obj_trajs']
        batch_size, num_agents, seq_len, features = obj_trajs.shape
        
        # 简单处理输入
        x = obj_trajs.view(batch_size, -1)
        output = self.network(x)
        
        # 构造预测输出
        predicted_trajectory = torch.randn(batch_size, 6, 60, 5)  # (batch, modes, time, features)
        predicted_probability = torch.softmax(torch.randn(batch_size, 6), dim=1)  # (batch, modes)
        
        prediction = {
            'predicted_trajectory': predicted_trajectory,
            'predicted_probability': predicted_probability
        }
        
        # 简单损失计算
        loss = torch.mean(output ** 2)
        
        return prediction, loss

def load_balance_loss(routing_probs):
    """计算负载平衡损失"""
    expert_mean = routing_probs.mean(dim=0)
    loss = (expert_mean * routing_probs.sum(dim=0)).sum() / routing_probs.size(0)
    return loss

def train_router_only(num_epochs=10, batch_size=4):
    """只训练路由选择器"""
    print("Starting Router-only Training...")
    
    # 导入路由选择器
    from moe import TrajAttentionRouter
    
    # 创建路由选择器
    router = TrajAttentionRouter(config)
    
    # 创建优化器
    optimizer = optim.Adam(router.parameters(), lr=config['learning_rate'])
    
    # 创建模拟专家
    expert1 = SimpleExpert()
    expert2 = SimpleExpert()
    
    # 创建日志文件
    log_file = 'router_training_log.csv'
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'batch', 'total_loss', 'routing_loss', 'lb_loss', 'routing_probs'])
    
    # 训练循环
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 5  # 每个epoch的批次数量
        
        for batch_idx in range(num_batches):
            # 创建批次数据
            batch = create_dummy_batch(batch_size)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 路由选择器前向传播
            routing_probs = router(batch)
            
            # 模拟专家损失 (这里我们使用固定的目标损失来测试路由是否能学习)
            # 假设专家1在当前批次上表现更好，路由应该学会选择专家1
            expert1_loss = torch.tensor(1.5)  # 较好的损失
            expert2_loss = torch.tensor(3.0)  # 较差的损失
            
            # 根据路由概率计算加权损失
            weighted_loss = (routing_probs[:, 0] * expert1_loss + routing_probs[:, 1] * expert2_loss).mean()
            
            # 计算负载平衡损失
            lb_loss_val = load_balance_loss(routing_probs)
            lb_loss_weight = 0.01
            
            # 总损失
            total_loss = weighted_loss + lb_loss_weight * lb_loss_val
            
            # 反向传播
            total_loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累计损失
            epoch_loss += total_loss.item()
            
            # 记录日志
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, 
                    batch_idx, 
                    total_loss.item(), 
                    weighted_loss.item(), 
                    lb_loss_val.item(),
                    routing_probs.detach().cpu().numpy().tolist()
                ])
            
            # 打印进度
            if batch_idx % 2 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{num_batches}], "
                      f"Loss: {total_loss.item():.4f}, Routing: {routing_probs.mean(dim=0).detach().cpu().numpy()}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        # 打印当前路由概率分布
        dummy_batch = create_dummy_batch(100)  # 更大的批次来观察平均行为
        with torch.no_grad():
            routing_probs = router(dummy_batch)
            mean_probs = routing_probs.mean(dim=0)
            print(f"Average routing probabilities: {mean_probs.cpu().numpy()}")
    
    print("Training completed!")
    print(f"Log saved to {log_file}")

def test_router_convergence():
    """测试路由选择器是否能收敛到正确的专家选择"""
    print("\nTesting Router Convergence...")
    
    # 导入路由选择器
    from moe import TrajAttentionRouter
    
    # 创建路由选择器
    router = TrajAttentionRouter(config)
    
    # 创建优化器
    optimizer = optim.Adam(router.parameters(), lr=0.01)
    
    # 创建固定的目标路由概率 (例如：总是选择专家1)
    target_probs = torch.tensor([[1.0, 0.0]])  # 总是选择专家1
    
    # 训练路由选择器来匹配目标概率
    print("Training router to match target probabilities...")
    
    for epoch in range(100):
        optimizer.zero_grad()
        
        # 创建批次数据
        batch = create_dummy_batch(16)
        
        # 路由选择器前向传播
        routing_probs = router(batch)
        
        # 计算与目标概率的KL散度损失
        target_expanded = target_probs.expand(routing_probs.shape[0], -1)
        kl_loss = torch.nn.functional.kl_div(
            torch.log(routing_probs + 1e-8), 
            target_expanded, 
            reduction='batchmean'
        )
        
        # 反向传播
        kl_loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 打印进度
        if epoch % 20 == 0:
            mean_probs = routing_probs.mean(dim=0)
            print(f"Epoch {epoch}, KL Loss: {kl_loss.item():.4f}, "
                  f"Mean routing probs: {mean_probs.detach().cpu().numpy()}")
    
    # 最终测试
    batch = create_dummy_batch(100)
    with torch.no_grad():
        routing_probs = router(batch)
        mean_probs = routing_probs.mean(dim=0)
        print(f"Final mean routing probabilities: {mean_probs.cpu().numpy()}")
        
        # 检查是否收敛
        if mean_probs[0] > 0.8:
            print("✅ Router successfully converged to target distribution!")
        else:
            print("⚠️ Router did not converge properly.")

if __name__ == "__main__":
    print("Debugging MOE Router...")
    print("=" * 50)
    
    # 测试路由训练
    train_router_only(num_epochs=5)
    
    # 测试路由收敛性
    test_router_convergence()
    
    print("\n" + "=" * 50)
    print("🎉 Debugging completed!")