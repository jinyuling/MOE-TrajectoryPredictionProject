#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版MOE模型 - 用于演示和调试
"""

class SimpleRouter:
    """简化版路由选择器"""
    
    def __init__(self, num_experts=2):
        self.num_experts = num_experts
        print(f"🔧 创建路由选择器，专家数量: {num_experts}")
    
    def forward(self, batch_data):
        """模拟前向传播"""
        batch_size = len(batch_data)
        import random
        
        # 模拟路由概率（随机生成，实际应该基于输入数据计算）
        routing_probs = []
        for _ in range(batch_size):
            # 生成随机概率并归一化
            probs = [random.random() for _ in range(self.num_experts)]
            total = sum(probs)
            probs = [p/total for p in probs]
            routing_probs.append(probs)
        
        print(f"📊 批次大小: {batch_size}, 路由概率: {routing_probs}")
        return routing_probs

class SimpleExpert:
    """简化版专家模型"""
    
    def __init__(self, expert_id):
        self.expert_id = expert_id
        print(f"🔧 创建专家模型 {expert_id}")
    
    def forward(self, batch_data):
        """模拟前向传播"""
        batch_size = len(batch_data)
        # 模拟预测结果
        import random
        
        # 模拟预测轨迹
        predicted_trajectory = [[[random.random() for _ in range(2)] 
                                for _ in range(60)] for _ in range(6)]
        
        # 模拟预测概率
        predicted_probability = [random.random() for _ in range(6)]
        total = sum(predicted_probability)
        predicted_probability = [p/total for p in predicted_probability]
        
        prediction = {
            'predicted_trajectory': predicted_trajectory,
            'predicted_probability': predicted_probability
        }
        
        # 模拟损失
        loss = random.random() * 5  # 随机损失值
        
        print(f"🤖 专家 {self.expert_id} 处理完成，损失: {loss:.4f}")
        return prediction, loss

class SimpleMOE:
    """简化版MOE模型"""
    
    def __init__(self, num_experts=2):
        print("🔧 创建MOE模型...")
        self.num_experts = num_experts
        self.router = SimpleRouter(num_experts)
        self.experts = [SimpleExpert(i) for i in range(num_experts)]
        print("✅ MOE模型创建完成")
    
    def forward(self, batch_data):
        """MOE前向传播"""
        print("\n🔄 MOE前向传播开始...")
        
        # 1. 路由选择器决策
        print("1️⃣ 路由选择器决策...")
        routing_probs = self.router.forward(batch_data)
        
        # 2. 专家模型处理
        print("2️⃣ 专家模型处理...")
        expert_outputs = []
        expert_losses = []
        
        for i, expert in enumerate(self.experts):
            print(f"   处理专家 {i}...")
            output, loss = expert.forward(batch_data)
            expert_outputs.append(output)
            expert_losses.append(loss)
        
        # 3. 根据路由概率组合结果
        print("3️⃣ 组合专家结果...")
        batch_size = len(batch_data)
        
        # 简化的组合逻辑
        final_output = expert_outputs[0]  # 简单使用第一个专家的结果
        weighted_loss = sum(routing_probs[0][i] * expert_losses[i] 
                           for i in range(self.num_experts))
        
        print(f"✅ MOE前向传播完成，加权损失: {weighted_loss:.4f}")
        return final_output, weighted_loss, routing_probs

def create_dummy_data(batch_size=4):
    """创建模拟数据"""
    print(f"📦 创建模拟批次数据，批次大小: {batch_size}")
    dummy_data = [f"sample_{i}" for i in range(batch_size)]
    return dummy_data

def demo_moe():
    """演示MOE模型"""
    print("🚀 简化版MOE模型演示")
    print("=" * 40)
    
    # 创建MOE模型
    moe = SimpleMOE(num_experts=2)
    
    # 创建模拟数据
    batch_data = create_dummy_data(batch_size=3)
    
    # 前向传播
    output, loss, routing_probs = moe.forward(batch_data)
    
    print("\n" + "=" * 40)
    print("🎉 MOE模型演示完成")
    print(f"最终损失: {loss:.4f}")
    print(f"路由概率: {routing_probs[0]}")

if __name__ == "__main__":
    demo_moe()