#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整优化训练脚本 - 展示MOE模型的所有改进
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from optimized_moe import OptimizedMOE
import numpy as np

class TrajectoryDataset(Dataset):
    """轨迹预测数据集"""
    
    def __init__(self, size=100, seq_len=20, num_agents=30, feature_dim=32):
        self.size = size
        self.seq_len = seq_len
        self.num_agents = num_agents
        self.feature_dim = feature_dim
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 创建模拟轨迹数据
        obj_trajs = torch.randn(self.num_agents, self.seq_len, self.feature_dim)
        obj_trajs_mask = torch.ones(self.num_agents, self.seq_len)
        
        # 随机遮蔽一些代理
        mask_indices = torch.rand(self.num_agents) > 0.7
        obj_trajs_mask[mask_indices] = 0
        
        input_dict = {
            'obj_trajs': obj_trajs,
            'obj_trajs_mask': obj_trajs_mask,
            'map_polylines': torch.randn(100, 20, 64),
            'map_polylines_mask': torch.ones(100, 20),
            'track_index_to_predict': torch.randint(0, self.num_agents, (1,)),
            'center_gt_trajs': torch.randn(60, 2),
            'center_gt_trajs_mask': torch.ones(60),
            'center_gt_final_valid_idx': torch.randint(0, 60, (1,)),
            'center_objects_world': torch.randn(7),
            'map_center': torch.randn(2),
            'scenario_id': f"scenario_{idx}",
            'center_objects_id': f"object_{idx}",
            'center_objects_type': torch.randint(1, 4, (1,)),
            'dataset_name': "dummy"
        }
        
        return {'input_dict': input_dict}
    
    def collate_fn(self, batch):
        """批处理函数"""
        batched_input_dict = {}
        keys = batch[0]['input_dict'].keys()
        
        for key in keys:
            values = [sample['input_dict'][key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                batched_input_dict[key] = torch.stack(values)
            else:
                batched_input_dict[key] = values
        
        return {
            'batch_size': len(batch),
            'input_dict': batched_input_dict
        }

class OptimizedMOETrainingModule(pl.LightningModule):
    """优化版MOE训练模块"""
    
    def __init__(self, model, learning_rate=0.001):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # MOE模型前向传播
        prediction, total_loss, routing_probs = self.model(batch)
        
        # 记录损失和路由统计信息
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True)
        
        # 记录路由统计信息
        stats = self.model.get_routing_statistics(routing_probs)
        self.log('train_expert_usage_var', stats['usage_variance'], on_step=False, on_epoch=True)
        self.log('train_max_expert_usage', stats['max_usage'], on_step=False, on_epoch=True)
        self.log('train_min_expert_usage', stats['min_usage'], on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # MOE模型前向传播
        prediction, total_loss, routing_probs = self.model(batch)
        
        # 记录损失和路由统计信息
        self.log('val_total_loss', total_loss, on_step=False, on_epoch=True)
        
        # 记录路由统计信息
        stats = self.model.get_routing_statistics(routing_probs)
        self.log('val_expert_usage_var', stats['usage_variance'], on_step=False, on_epoch=True)
        self.log('val_max_expert_usage', stats['max_usage'], on_step=False, on_epoch=True)
        self.log('val_min_expert_usage', stats['min_usage'], on_step=False, on_epoch=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]

def run_optimization_demo():
    """运行优化演示"""
    print("🚀 开始MOE模型完整优化演示...")
    
    # 配置参数
    config = {
        'input_dim': 32,
        'num_experts': 3,  # 增加专家数量
        'hidden_dim': 64,
        'output_dim': 32
    }
    
    # 创建优化版MOE模型
    print("🔧 创建优化版MOE模型...")
    moe_model = OptimizedMOE(config)
    
    # 创建训练模块
    training_module = OptimizedMOETrainingModule(moe_model, learning_rate=0.001)
    
    # 创建数据集
    print("📦 创建数据集...")
    train_dataset = TrajectoryDataset(200)  # 增加训练数据量
    val_dataset = TrajectoryDataset(40)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,  # 增加批次大小
        shuffle=True, 
        collate_fn=train_dataset.collate_fn,
        num_workers=0  # 避免Windows多进程问题
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False, 
        collate_fn=val_dataset.collate_fn,
        num_workers=0
    )
    
    # 创建训练器
    print("🏃 创建训练器...")
    trainer = pl.Trainer(
        max_epochs=5,  # 增加训练轮数
        accelerator="cpu",  # 使用CPU训练
        devices=1,
        logger=False,  # 禁用日志记录
        enable_checkpointing=False  # 禁用检查点
    )
    
    # 开始训练
    print("🔥 开始训练...")
    trainer.fit(
        model=training_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # 展示最终的路由统计信息
    print("📊 最终路由统计信息:")
    # 创建一个示例批次进行前向传播
    sample_batch = next(iter(val_loader))
    _, _, routing_probs = moe_model(sample_batch)
    final_stats = moe_model.get_routing_statistics(routing_probs)
    
    print(f"   专家使用方差: {final_stats['usage_variance'].item():.6f}")
    print(f"   最大专家使用率: {final_stats['max_usage'].item():.4f}")
    print(f"   最小专家使用率: {final_stats['min_usage'].item():.4f}")
    print(f"   专家使用率: {final_stats['expert_usage']}")
    
    print("🎉 MOE模型完整优化演示完成!")

def compare_with_baseline():
    """与基线模型对比"""
    print("\n🔍 与基线模型对比...")
    
    # 基线配置（2个专家）
    baseline_config = {
        'input_dim': 32,
        'num_experts': 2,
        'hidden_dim': 64,
        'output_dim': 32
    }
    
    # 优化配置（3个专家）
    optimized_config = {
        'input_dim': 32,
        'num_experts': 3,
        'hidden_dim': 64,
        'output_dim': 32
    }
    
    # 创建模型
    baseline_model = OptimizedMOE(baseline_config)
    optimized_model = OptimizedMOE(optimized_config)
    
    # 创建相同输入
    batch_size = 4
    input_dict = {
        'input_dict': {
            'obj_trajs': torch.randn(batch_size, 30, 20, 32)
        }
    }
    
    # 前向传播
    _, baseline_loss, baseline_routing = baseline_model(input_dict)
    _, optimized_loss, optimized_routing = optimized_model(input_dict)
    
    # 比较结果
    print(f"   基线模型损失: {baseline_loss.item():.4f}")
    print(f"   优化模型损失: {optimized_loss.item():.4f}")
    print(f"   损失改善: {baseline_loss.item() - optimized_loss.item():.4f}")
    
    # 比较路由统计
    baseline_stats = baseline_model.get_routing_statistics(baseline_routing)
    optimized_stats = optimized_model.get_routing_statistics(optimized_routing)
    
    print(f"   基线专家使用方差: {baseline_stats['usage_variance'].item():.6f}")
    print(f"   优化专家使用方差: {optimized_stats['usage_variance'].item():.6f}")
    print(f"   负载平衡改善: {baseline_stats['usage_variance'].item() - optimized_stats['usage_variance'].item():.6f}")
    
    print("✅ 对比完成!")

if __name__ == "__main__":
    # 运行优化演示
    run_optimization_demo()
    
    # 与基线模型对比
    compare_with_baseline()