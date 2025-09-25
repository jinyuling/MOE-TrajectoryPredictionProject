import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np
from torch import optim
from torch.distributions import MultivariateNormal, Laplace
from torch.optim.lr_scheduler import MultiStepLR

# 从本地导入而不是unitraj
from base_model import BaseModel
from unitraj.models.wayformer.wayformer_utils import PerceiverEncoder
# from models import build_model

class TrajAttentionRouter(BaseModel):
    def __init__(self, config):
        super(TrajAttentionRouter, self).__init__(config)
        
        # 确保config是字典
        if not isinstance(config, dict):
            config = {}
        
        self.config = config
        self.d_k = config.get('hidden_size', 128)
        self.past_T = config.get('past_len', 20)
        self.map_attr = config.get('num_map_feature', 64)
        self.k_attr = config.get('num_agent_feature', 32)
        self.num_queries_enc = config.get('num_queries_enc', 16)
        self.num_queries_dec = config.get('num_queries_dec', 16)
        self.max_num_roads = config.get('max_num_roads', 100)
        self.num_experts = config.get('num_experts', 2)
        self._M = config.get('max_num_agents', 30)
        
        # 特征提取器 - 改进的轨迹特征提取
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
        
        # 注意力机制
        self.perceiver_encoder = PerceiverEncoder(192, self.d_k,
                                                 num_cross_attention_qk_channels=self.d_k,
                                                  num_cross_attention_v_channels=self.d_k,
                                                  num_self_attention_qk_channels=self.d_k,
                                                  num_self_attention_v_channels=self.d_k)
        
        # 路由决策网络
        self.routing_head = nn.Sequential(
            nn.Linear(self.d_k, self.d_k // 2),
            nn.ReLU(),
            nn.LayerNorm(self.d_k // 2),
            nn.Linear(self.d_k // 2, self.num_experts)
        )
        
        # 位置编码
        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, (self._M + 1), self.d_k)),
            requires_grad=True
        )
        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, self.past_T, 1, self.d_k)),
            requires_grad=True
        )
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def process_observations(self, ego, agents):
        '''
        :param observations: (B, T, N+2, A+1) where N+2 is [ego, other_agents, env]
        :return: a tensor of only the agent dynamic states, active_agent masks and env masks.
        '''
        # ego stuff
        ego_tensor = ego[:, :, :self.k_attr]
        env_masks_orig = ego[:, :, -1]
        env_masks = (1.0 - env_masks_orig).to(torch.bool)
        env_masks = env_masks.unsqueeze(1).repeat(1, self.num_queries_dec, 1).view(ego.shape[0] * self.num_queries_dec,
                                                                                   -1)

        # Agents stuff
        temp_masks = torch.cat((torch.ones_like(env_masks_orig.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).to(torch.bool)  # only for agents.
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states

        return ego_tensor, opps_tensor, opps_masks, env_masks

    def extract_trajectory_features(self, agents_in):
        """提取轨迹特征"""
        B, T, N, D = agents_in.shape
        # 展平并编码轨迹特征 - 使用reshape而不是view
        agents_flat = agents_in.reshape(B*T*N, D)
        traj_features = self.trajectory_encoder(agents_flat)
        return traj_features.reshape(B, T, N, -1)
    
    def extract_road_features(self, roads):
        """提取路网特征"""
        B, R, P, D = roads.shape
        # 展平并编码路网特征 - 使用reshape而不是view
        roads_flat = roads.reshape(B*R*P, D)
        road_features = self.road_encoder(roads_flat)
        return road_features.reshape(B, R, P, -1)

    def forward(self, x):
        inputs = x['input_dict']
        agents_in, agents_mask, roads = inputs['obj_trajs'], inputs['obj_trajs_mask'], inputs['map_polylines']
        ego_in = torch.gather(agents_in, 1, inputs['track_index_to_predict'].view(-1, 1, 1, 1).repeat(1, 1,
                                                                                                      *agents_in.shape[
                                                                                                       -2:])).squeeze(1)
        ego_mask = torch.gather(agents_mask, 1, inputs['track_index_to_predict'].view(-1, 1, 1).repeat(1, 1,
                                                                                                       agents_mask.shape[
                                                                                                           -1])).squeeze(
            1)
        agents_in = torch.cat([agents_in, agents_mask.unsqueeze(-1)], dim=-1)
        agents_in = agents_in.transpose(1, 2)
        ego_in = torch.cat([ego_in, ego_mask.unsqueeze(-1)], dim=-1)
        roads = torch.cat([inputs['map_polylines'], inputs['map_polylines_mask'].unsqueeze(-1)], dim=-1)

        B = ego_in.size(0)
        num_agents = agents_in.shape[2] + 1
        
        # 提取轨迹特征
        traj_features = self.extract_trajectory_features(agents_in[:, :, :, :self.k_attr])
        
        # 提取路网特征
        road_features = self.extract_road_features(roads[:, :self.max_num_roads, :, :self.map_attr])
        
        # 合并特征
        # 轨迹特征处理
        ego_tensor, _agents_tensor, opps_masks_agents, env_masks = self.process_observations(ego_in, agents_in)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2)
        
        # 添加位置编码 - 确保维度匹配
        agents_pos_emb = self.agents_positional_embedding[:, :, :min(num_agents, self.agents_positional_embedding.size(2))]
        temporal_pos_emb = self.temporal_positional_embedding
        
        # 如果维度不匹配，进行填充或裁剪
        if agents_pos_emb.size(2) < num_agents:
            padding = torch.zeros(B, 1, num_agents - agents_pos_emb.size(2), self.d_k)
            agents_pos_emb = torch.cat([agents_pos_emb, padding], dim=2)
        elif agents_pos_emb.size(2) > num_agents:
            agents_pos_emb = agents_pos_emb[:, :, :num_agents, :]
        
        agents_emb = traj_features + agents_pos_emb + temporal_pos_emb
        agents_emb = agents_emb.view(B, -1, self.d_k)
        
        # 路网特征处理
        road_pts_feats = road_features.view(B, -1, self.d_k)
        opps_masks_roads = (1.0 - roads[:, :self.max_num_roads, :, -1]).to(torch.bool)
        mixed_input_masks = torch.concat([opps_masks_agents.view(B, -1), opps_masks_roads.view(B, -1)], dim=1)
        
        # 合并所有输入特征
        mixed_input_features = torch.concat([agents_emb, road_pts_feats], dim=1)
        
        # 通过Perceiver编码器
        context = self.perceiver_encoder(mixed_input_features, mixed_input_masks)
        
        # 全局池化获取场景表示
        pooled = context.mean(dim=1)
        
        # 路由决策
        routing_logits = self.routing_head(pooled)
        return F.softmax(routing_logits, dim=-1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], eps=0.0001, weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=self.config['learning_rate_sched'], gamma=0.5, verbose=True)
        return [optimizer], [scheduler]


class MOE(BaseModel):
    def __init__(self, config, init_cfg=None):
        super(MOE, self).__init__(config)
        
        # 基本配置 - 确保config是字典
        if isinstance(config, str):
            # 如果config是字符串，创建一个默认配置
            config = {
                'train_router_only': True,
                'num_experts': 2,
                'router': {'k': 2},
                'experts_cfg': []
            }
        
        self.train_router_only = config.get('train_router_only', True) if isinstance(config, dict) else True
        self.k = config.get('router', {}).get('k', 2) if isinstance(config, dict) else 2
        self.config = config
        self.num_experts = config.get('num_experts', 2) if isinstance(config, dict) else 2
        
        # 为简化演示，创建两个简单的线性模型作为专家
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            ),
            nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
        ])
        
        # 创建一个简化的路由器
        self.router = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_experts),
            nn.Softmax(dim=-1)
        )
        
        # 冻结专家参数（如果需要）
        if self.train_router_only:
            self.freeze_experts()
    
    def freeze_experts(self):
        """冻结所有专家的参数"""
        for idx, expert in enumerate(self.experts):
            for param in expert.parameters():
                param.requires_grad = False
            print(f"🔒 Expert {idx} parameters frozen")
    
    def forward(self, x):
        # 简化版前向传播
        inputs = x['input_dict']
        
        # 获取输入特征（简化处理）
        # 假设我们使用obj_trajs的第一个代理的最后一个时间步作为输入
        input_features = inputs['obj_trajs'][:, 0, -1, :32]  # 取前32个特征
        
        # 路由器决策
        routing_probs = self.router(input_features)
        
        # 专家处理
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(input_features)
            expert_outputs.append(expert_output)
        
        # 根据路由概率组合结果
        batch_size = input_features.shape[0]
        final_output = torch.zeros_like(expert_outputs[0])
        
        for i in range(self.num_experts):
            final_output += routing_probs[:, i:i+1] * expert_outputs[i]
        
        # 创建预测结果字典
        prediction = {
            'predicted_trajectory': torch.randn(batch_size, 6, 60, 5),  # 模拟轨迹预测
            'predicted_probability': torch.softmax(torch.randn(batch_size, 6), dim=-1)  # 模拟概率预测
        }
        
        # 计算损失（简化版）
        loss = torch.mean(final_output ** 2)
        
        return prediction, loss, routing_probs
    
    def split_batch(self, batch):
        """分割批次"""
        input_dict = batch['input_dict']
        bs = next(iter(input_dict.values())).shape[0]
        return [{k: v[i:i+1] for k, v in input_dict.items()} for i in range(bs)]
    
    def merge_splits(self, splits, idx):
        """合并批次"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        def cat_vals(vals):
            v0 = vals[0]
            if isinstance(v0, torch.Tensor):
                return torch.cat(vals, dim=0)
            elif isinstance(v0, np.ndarray):
                return np.concatenate(vals, axis=0)
            elif isinstance(v0, list):
                return [cat_vals([v[j] for v in vals]) for j in range(len(v0))]
            else:
                return vals[0]
        
        merged = {}
        for k in splits[0].keys():
            vals = [splits[i][k] for i in idx]
            merged[k] = cat_vals(vals)
        
        return {
            'batch_size': len(idx),
            'input_dict': merged,
            'batch_sample_count': len(idx)
        }
    
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
        
        scheduler = MultiStepLR(optimizer, milestones=self.config['learning_rate_sched'], gamma=0.5, verbose=True)
        return [optimizer], [scheduler]
    
    def compute_loss(self, expert_output, batch):
        """计算专家输出的损失"""
        inputs = batch['input_dict']
        gt_traj = inputs['center_gt_trajs'].unsqueeze(1)
        gt_traj_mask = inputs['center_gt_trajs_mask'].unsqueeze(1)
        center_gt_final_valid_idx = inputs['center_gt_final_valid_idx']
        
        predicted_traj = expert_output['predicted_trajectory']
        predicted_prob = expert_output['predicted_probability']
        
        # 计算ADE损失
        ade_diff = torch.norm(predicted_traj[:, :, :, :2] - gt_traj[:, :, :, :2], 2, dim=-1)
        ade_losses = torch.sum(ade_diff * gt_traj_mask, dim=-1) / torch.sum(gt_traj_mask, dim=-1)
        
        # 计算FDE损失
        bs, modes, future_len = ade_diff.shape
        center_gt_final_valid_idx = center_gt_final_valid_idx.view(-1, 1, 1).repeat(1, modes, 1).to(torch.int64)
        fde = torch.gather(ade_diff, -1, center_gt_final_valid_idx).squeeze(-1)
        
        # 结合概率和损失
        weighted_ade = torch.sum(ade_losses * predicted_prob, dim=1)
        weighted_fde = torch.sum(fde * predicted_prob, dim=1)
        
        return torch.mean(weighted_ade + weighted_fde)


def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# 封装了MoE模型
class MoENetwork(BaseModel):
    def __init__(self, num_experts, in_channels, experts_cfg, router, init_cfg=None):
        super(MoENetwork, self).__init__(init_cfg)
        self.moe = MOE(num_experts, in_channels, experts_cfg, router)

        self.lb_loss_weight_initial = 0.01
        self.lb_loss_decay_rate = 0.95
        self.lb_loss_weight = self.lb_loss_weight_initial
    
    def forward(self, x):
        return self.moe(x) 
    # 取路由概率
    def get_routing_probs(self):
        return self.moe.last_routing_probs
    # 包含负载平衡(load balancing)损失权重更新
    def update_lb_loss_weight(self, epoch):
        self.lb_loss_weight = self.lb_loss_weight_initial * (self.lb_loss_decay_rate ** epoch)
        print(f"lb_loss_weight updated to {self.lb_loss_weight}")
    
    def load_balance_loss(self, routing_probs):
        expert_mean = routing_probs.mean(dim=0)
        loss = (expert_mean * routing_probs.sum(dim=0)).sum()
        return loss