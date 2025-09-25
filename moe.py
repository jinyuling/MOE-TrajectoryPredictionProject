import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np
from torch import optim
from torch.distributions import MultivariateNormal, Laplace
from torch.optim.lr_scheduler import MultiStepLR

# from unitraj.models.moe.router import *
# from unitraj.models.moe.model_dapter import *
from unitraj.models.base_model.base_model import BaseModel
from unitraj.models.wayformer.wayformer_utils import PerceiverEncoder
# from models import build_model

class TrajAttentionRouter(BaseModel):
    def __init__(self, config):
        super(TrajAttentionRouter, self).__init__(config)
        self.config = config
        self.d_k = config['hidden_size']
        self.past_T = config['past_len']
        self.map_attr = config['num_map_feature']
        self.k_attr = config['num_agent_feature']
        self.num_queries_enc = config['num_queries_enc']
        self.num_queries_dec = config['num_queries_dec']
        self.max_num_roads = config['max_num_roads']
        self.num_experts = config['num_experts']
        self._M = config['max_num_agents'] 
        
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
        # 展平并编码轨迹特征
        agents_flat = agents_in.view(B*T*N, D)
        traj_features = self.trajectory_encoder(agents_flat)
        return traj_features.view(B, T, N, -1)
    
    def extract_road_features(self, roads):
        """提取路网特征"""
        B, R, P, D = roads.shape
        # 展平并编码路网特征
        roads_flat = roads.view(B*R*P, D)
        road_features = self.road_encoder(roads_flat)
        return road_features.view(B, R, P, -1)

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
        
        # 添加位置编码
        agents_emb = traj_features + self.agents_positional_embedding[:, :, :num_agents] + self.temporal_positional_embedding
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
        from models import build_model
        super(MOE, self).__init__(config)
        
        # 基本配置
        self.train_router_only = config.get('train_router_only', True)
        self.k = config.router['k']
        self.config = config
        self.num_experts = config['num_experts']
        
        # 构建专家和路由器
        self.experts = nn.ModuleList([build_model(cfg) for cfg in config.experts_cfg])
        self.router = TrajAttentionRouter(config)
        
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
        # 获取路由概率
        routing_probs = self.router(x)
        self.last_routing_probs = routing_probs
        B = routing_probs.size(0)
        
        # 选择专家
        if self.training:
            weights = routing_probs
            indices = torch.arange(self.num_experts, device=routing_probs.device).repeat(B, 1)
        else:
            weights, indices = torch.topk(routing_probs, k=self.k, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # 初始化输出
        expert_output_predicted_trajectory = torch.zeros((B, 6, 60, 5), device=routing_probs.device)
        expert_output_predicted_probability = torch.zeros((B, 6), device=routing_probs.device)
        
        # 专家前向传播（注意：在训练路由时不需要梯度，在联合训练时需要）
        with torch.set_grad_enabled(not self.train_router_only):
            for i, expert in enumerate(self.experts):
                idx, top = torch.where(indices == i)
                if idx.numel() == 0:
                    continue
                
                # 准备批次
                splits = self.split_batch(x)
                new_batch = self.merge_splits(splits, idx)
                
                # 专家前向传播
                expert_output, _ = expert(new_batch)
                
                # 收集输出
                expert_output_predicted_trajectory[idx] = expert_output['predicted_trajectory']
                expert_output_predicted_probability[idx] = expert_output['predicted_probability']
        
        # 应用路由权重
        expert_output = {
            'predicted_trajectory': expert_output_predicted_trajectory,
            'predicted_probability': expert_output_predicted_probability
        }
        
        # 计算损失
        expert_output_Loss = self.compute_loss(expert_output, x) if hasattr(self, 'compute_loss') else torch.tensor(0.0)
        
        return expert_output, expert_output_Loss, routing_probs
    
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