"""
简化版PerceiverEncoder实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceiverEncoder(nn.Module):
    """简化版Perceiver编码器"""
    
    def __init__(self, num_latents, latent_dim, 
                 num_cross_attention_qk_channels=None,
                 num_cross_attention_v_channels=None,
                 num_self_attention_qk_channels=None,
                 num_self_attention_v_channels=None):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        
        # 创建可学习的潜在变量
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        
        # 简化的交叉注意力
        self.cross_attention = SimpleAttention(
            q_dim=latent_dim,
            k_dim=num_cross_attention_qk_channels or latent_dim,
            v_dim=num_cross_attention_v_channels or latent_dim,
            out_dim=latent_dim
        )
        
        # 简化的自注意力
        self.self_attention = SimpleAttention(
            q_dim=latent_dim,
            k_dim=num_self_attention_qk_channels or latent_dim,
            v_dim=num_self_attention_v_channels or latent_dim,
            out_dim=latent_dim
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(latent_dim)
        self.layer_norm2 = nn.LayerNorm(latent_dim)
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入特征 (batch_size, seq_len, feature_dim)
            mask: 注意力掩码 (batch_size, seq_len)
            
        Returns:
            编码后的特征 (batch_size, num_latents, latent_dim)
        """
        batch_size = x.shape[0]
        
        # 初始化潜在变量
        latents = self.latents.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 交叉注意力：潜在变量关注输入
        latents = self.layer_norm1(latents + self.cross_attention(latents, x, x, mask))
        
        # 自注意力：潜在变量内部交互
        latents = self.layer_norm2(latents + self.self_attention(latents, latents, latents))
        
        return latents

class SimpleAttention(nn.Module):
    """简化的注意力机制"""
    
    def __init__(self, q_dim, k_dim, v_dim, out_dim):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, out_dim)
        self.k_proj = nn.Linear(k_dim, out_dim)
        self.v_proj = nn.Linear(v_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        
    def forward(self, q, k, v, mask=None):
        """
        注意力计算
        
        Args:
            q: 查询 (batch_size, seq_len_q, q_dim)
            k: 键 (batch_size, seq_len_k, k_dim)
            v: 值 (batch_size, seq_len_k, v_dim)
            mask: 掩码 (batch_size, seq_len_k)
            
        Returns:
            注意力输出 (batch_size, seq_len_q, out_dim)
        """
        # 投影
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        
        # 应用掩码
        if mask is not None:
            # 扩展掩码维度以匹配注意力分数
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len_k)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力权重
        weights = F.softmax(scores, dim=-1)
        
        # 计算输出
        output = torch.matmul(weights, v)
        output = self.out_proj(output)
        
        return output