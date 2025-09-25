#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版训练脚本 - 用于演示MOE训练流程
"""

import random
import time

class SimpleTrainer:
    """简化版训练器"""
    
    def __init__(self, model, num_epochs=5):
        self.model = model
        self.num_epochs = num_epochs
        self.current_epoch = 0
        print(f"🔧 创建训练器，训练轮数: {num_epochs}")
    
    def create_dummy_batch(self, batch_size=4):
        """创建模拟批次数据"""
        return [f"sample_{i}" for i in range(batch_size)]
    
    def train_epoch(self):
        """训练一个epoch"""
        print(f"\n📈 训练 Epoch {self.current_epoch + 1}/{self.num_epochs}")
        
        # 模拟训练批次
        num_batches = 3
        epoch_loss = 0.0
        
        for batch_idx in range(num_batches):
            # 创建批次数据
            batch_data = self.create_dummy_batch(batch_size=4)
            
            # 模型前向传播
            try:
                output, loss, routing_probs = self.model.forward(batch_data)
                epoch_loss += loss
                
                # 显示批次信息
                print(f"   Batch {batch_idx + 1}/{num_batches} - Loss: {loss:.4f}")
                
                # 模拟训练时间
                time.sleep(0.5)
            except Exception as e:
                print(f"   ❌ 批次训练出错: {e}")
                continue
        
        avg_loss = epoch_loss / num_batches
        print(f"   Epoch {self.current_epoch + 1} 平均损失: {avg_loss:.4f}")
        
        self.current_epoch += 1
        return avg_loss
    
    def train(self):
        """完整训练流程"""
        print("🚀 开始训练...")
        print("="*40)
        
        losses = []
        for epoch in range(self.num_epochs):
            loss = self.train_epoch()
            losses.append(loss)
            
            # 显示进度
            if epoch < self.num_epochs - 1:
                print(f"   ⏱️ 等待下一epoch...")
                time.sleep(1)
        
        print("\n" + "="*40)
        print("🎉 训练完成!")
        print(f"初始损失: {losses[0]:.4f}")
        print(f"最终损失: {losses[-1]:.4f}")
        print(f"损失下降: {losses[0] - losses[-1]:.4f}")
        
        if losses[-1] < losses[0]:
            print("✅ 损失成功下降!")
        else:
            print("⚠️ 损失未下降，请检查模型设置")
        
        return losses

def demo_training():
    """演示训练流程"""
    print("🎯 MOE模型训练演示")
    print("="*40)
    
    # 导入简化版MOE模型
    try:
        from simple_moe import SimpleMOE
        print("✅ 成功导入简化版MOE模型")
        
        # 创建MOE模型
        moe_model = SimpleMOE(num_experts=2)
        
        # 创建训练器
        trainer = SimpleTrainer(moe_model, num_epochs=3)
        
        # 开始训练
        losses = trainer.train()
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("💡 尝试运行 simple_moe.py 来验证模型是否正常工作")

if __name__ == "__main__":
    demo_training()