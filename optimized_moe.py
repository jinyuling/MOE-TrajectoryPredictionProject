#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版MOE模型 - 包含改进的路由选择器、专家集成策略和负载平衡机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OptimizedRouter(nn.Module):
    """优化版路由选择器"""
    
    def __init__(self, input_dim=32, num_experts=2, hidden_dim=64):
        super(OptimizedRouter, self).__init__()
        self.num_experts = num_experts
        
        # 简化版路由网络
        self.router_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # 可学习的温度参数，用于控制路由的锐度
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 (batch_size, feature_dim)
            
        Returns:
            路由概率 (batch_size, num_experts)
        """
        # 确保输入维度正确
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 路由决策
        routing_logits = self.router_network(x)
        
        # 应用温度缩放
        scaled_logits = routing_logits / self.temperature
        
        # 计算路由概率
        routing_probs = F.softmax(scaled_logits, dim=-1)
        
        return routing_probs

class LoadBalancingLoss(nn.Module):
    """负载平衡损失"""
    
    def __init__(self, weight=0.01):
        super(LoadBalancingLoss, self).__init__()
        self.weight = weight
    
    def forward(self, routing_probs):
        """
        计算负载平衡损失
        
        Args:
            routing_probs: 路由概率 (batch_size, num_experts)
            
        Returns:
            负载平衡损失
        """
        # 计算每个专家的平均使用概率
        expert_mean = routing_probs.mean(dim=0)
        
        # 计算负载平衡损失 - 鼓励专家均匀使用
        lb_loss = torch.var(expert_mean) * self.weight
        
        return lb_loss

class OptimizedMOE(nn.Module):
    """优化版MOE模型"""
    
    def __init__(self, config):
        super(OptimizedMOE, self).__init__()
        
        # 配置参数
        self.input_dim = config.get('input_dim', 32)
        self.num_experts = config.get('num_experts', 2)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.output_dim = config.get('output_dim', 32)
        
        # 创建专家模型
        self.experts = nn.ModuleList([
            self._create_expert() for _ in range(self.num_experts)
        ])
        
        # 创建优化版路由选择器
        self.router = OptimizedRouter(
            input_dim=self.input_dim,
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim
        )
        
        # 创建负载平衡损失模块
        self.load_balancing_loss = LoadBalancingLoss(weight=0.01)
        
        # 输出层
        self.output_layer = nn.Linear(self.output_dim, self.output_dim)
        
    def _create_expert(self):
        """创建专家模型"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征字典
            
        Returns:
            预测结果、总损失、路由概率
        """
        # 提取输入特征 - 确保维度匹配
        inputs = x['input_dict']
        input_features = inputs['obj_trajs'][:, 0, -1, :self.input_dim]  # 取前input_dim个特征
        
        # 确保输入特征维度正确
        if input_features.dim() == 1:
            input_features = input_features.unsqueeze(0)
        
        # 路由器决策
        routing_probs = self.router(input_features)
        
        # 专家处理
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(input_features)
            expert_outputs.append(expert_output)
        
        # 基于路由概率的加权集成
        batch_size = input_features.shape[0]
        integrated_output = torch.zeros(batch_size, self.output_dim)
        for i in range(self.num_experts):
            integrated_output += routing_probs[:, i:i+1] * expert_outputs[i]
        
        # 最终输出
        final_output = self.output_layer(integrated_output)
        
        # 创建预测结果
        prediction = {
            'predicted_trajectory': torch.randn(batch_size, 6, 60, 5),  # 模拟轨迹预测
            'predicted_probability': torch.softmax(torch.randn(batch_size, 6), dim=-1)  # 模拟概率预测
        }
        
        # 计算损失
        main_loss = torch.mean(final_output ** 2)
        lb_loss = self.load_balancing_loss(routing_probs)
        total_loss = main_loss + lb_loss
        
        return prediction, total_loss, routing_probs
    
    def get_routing_statistics(self, routing_probs):
        """
        获取路由统计信息
        
        Args:
            routing_probs: 路由概率
            
        Returns:
            统计信息字典
        """
        expert_usage = routing_probs.mean(dim=0)
        return {
            'expert_usage': expert_usage,
            'max_usage': torch.max(expert_usage),
            'min_usage': torch.min(expert_usage),
            'usage_variance': torch.var(expert_usage)
        }

def demo_optimized_moe():
    """演示优化版MOE模型"""
    print("🚀 演示优化版MOE模型...")
    
    # 配置参数
    config = {
        'input_dim': 32,
        'num_experts': 3,
        'hidden_dim': 64,
        'output_dim': 32
    }
    
    # 创建模型
    model = OptimizedMOE(config)
    
    # 创建模拟输入 - 确保维度匹配
    batch_size = 4
    input_dict = {
        'input_dict': {
            'obj_trajs': torch.randn(batch_size, 30, 20, 32)  # 确保最后一维是32
        }
    }
    
    # 前向传播
    prediction, total_loss, routing_probs = model(input_dict)
    
    # 显示结果
    print(f"✅ 前向传播完成")
    print(f"   损失: {total_loss.item():.4f}")
    print(f"   路由概率: {routing_probs}")
    
    # 显示路由统计信息
    stats = model.get_routing_statistics(routing_probs)
    print(f"📊 路由统计:")
    for key, value in stats.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value:.4f}")
    
    print("🎉 优化版MOE模型演示完成!")

if __name__ == "__main__":
    demo_optimized_moe()