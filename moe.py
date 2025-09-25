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
        

        
        self.perceiver_encoder = PerceiverEncoder(192, self.d_k,
                                                 num_cross_attention_qk_channels=self.d_k,
                                                  num_cross_attention_v_channels=self.d_k,
                                                  num_self_attention_qk_channels=self.d_k,
                                                  num_self_attention_v_channels=self.d_k)
        self.selu = nn.SELU(inplace=True) 

        self.mlp = nn.Sequential(
            nn.Linear(self.d_k, self.d_k),
            nn.PReLU(),

            nn.Linear(self.d_k, self.d_k // 2),
            nn.PReLU(),

            nn.Linear(self.d_k // 2, self.num_experts)
        )
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))
        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, (self._M + 1), self.d_k)),
            requires_grad=True
        )
        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, self.past_T, 1, self.d_k)),
            requires_grad=True
        )
        self.road_pts_lin = nn.Sequential(init_(nn.Linear(self.map_attr, self.d_k)))
    
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
        # Encode all input observations (k_attr --> d_k)
        ego_tensor, _agents_tensor, opps_masks_agents, env_masks = self.process_observations(ego_in, agents_in)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2)
        agents_emb = self.selu(self.agents_dynamic_encoder(agents_tensor))
        agents_emb = (agents_emb + self.agents_positional_embedding[:, :,
                                   :num_agents] + self.temporal_positional_embedding).view(B, -1, self.d_k)
        road_pts_feats = self.selu(self.road_pts_lin(roads[:, :self.max_num_roads, :, :self.map_attr]).view(B, -1,
                                                                                                            self.d_k))# + self.map_positional_embedding
        mixed_input_features = torch.concat([agents_emb, road_pts_feats], dim=1)
        opps_masks_roads = (1.0 - roads[:, :self.max_num_roads, :, -1]).to(torch.bool)
        mixed_input_masks = torch.concat([opps_masks_agents.view(B, -1), opps_masks_roads.view(B, -1)], dim=1)
        # Process through Wayformer's encoder

        context = self.perceiver_encoder(mixed_input_features, mixed_input_masks)                # [B,192,256]
        pooled = context.mean(dim=1)               # [B,256]
        out = self.mlp(pooled)                  # [B,num_experts]
        return F.softmax(out, dim=-1)             # [B,num_experts]
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], eps=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=self.config['learning_rate_sched'], gamma=0.5,
                                verbose=True)
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
         # print(routing_probs)
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
        
        # 使用 no_grad 包装专家前向传播（节省显存）
        with torch.no_grad():
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
            optimizer = optim.Adam(trainable_params, lr=self.config['learning_rate'], eps=0.0001)
            print(f"✅ Optimizing {len(trainable_params)} Router parameters only")
        else:
            # 优化所有参数
            optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], eps=0.0001)
            print(f"✅ Optimizing all parameters")
        
        scheduler = MultiStepLR(optimizer, milestones=self.config['learning_rate_sched'], gamma=0.5, verbose=True)
        return [optimizer], [scheduler]
    
    
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
    def load_balance_loss(routing_probs):
        expert_mean = routing_probs.mean(dim=0)
        loss = (expert_mean * routing_probs.sum(dim=0)).sum()
        return loss