#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOE模型演示训练脚本
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from moe import MOE, TrajAttentionRouter
import numpy as np

class SimpleDataset(Dataset):
    """简单数据集"""
    
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 创建模拟输入数据
        input_dict = {
            'obj_trajs': torch.randn(30, 20, 32),  # 30个代理，20个时间步，32个特征
            'obj_trajs_mask': torch.ones(30, 20),
            'map_polylines': torch.randn(100, 20, 64),  # 100条路，每条路20个点，64个特征
            'map_polylines_mask': torch.ones(100, 20),
            'track_index_to_predict': torch.randint(0, 30, (1,)),
            'center_gt_trajs': torch.randn(60, 2),  # 60个时间步，2个坐标
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

class MOETrainingModule(pl.LightningModule):
    """MOE训练模块"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # MOE模型前向传播
        prediction, loss, routing_probs = self.model(batch)
        
        # 记录损失
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # MOE模型前向传播
        prediction, loss, routing_probs = self.model(batch)
        
        # 记录损失
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def main():
    """主函数"""
    print("🚀 开始MOE模型演示训练...")
    
    # 创建配置
    config = {
        'hidden_size': 128,
        'past_len': 20,
        'num_map_feature': 64,
        'num_agent_feature': 32,
        'num_queries_enc': 16,
        'num_queries_dec': 16,
        'max_num_roads': 100,
        'num_experts': 2,
        'max_num_agents': 30,
        'learning_rate': 0.001,
        'learning_rate_sched': [5, 8]
    }
    
    # 创建MOE模型
    print("🔧 创建MOE模型...")
    moe_model = MOE(config)
    
    # 冻结专家参数
    print("🔒 冻结专家参数...")
    for expert in moe_model.experts:
        for param in expert.parameters():
            param.requires_grad = False
    print("✅ 专家参数冻结完成")
    
    # 创建训练模块
    training_module = MOETrainingModule(moe_model)
    
    # 创建数据集
    print("📦 创建数据集...")
    train_dataset = SimpleDataset(100)
    val_dataset = SimpleDataset(20)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=val_dataset.collate_fn
    )
    
    # 创建训练器
    print("🏃 创建训练器...")
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="cpu",  # 使用CPU训练
        devices=1,
        logger=False  # 禁用日志记录
    )
    
    # 开始训练
    print("🔥 开始训练...")
    trainer.fit(
        model=training_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    print("🎉 训练完成!")

if __name__ == "__main__":
    main()