"""
模拟数据集类，用于演示和测试
"""

import torch
import numpy as np
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    """模拟数据集"""
    
    def __init__(self, cfg, val=False):
        """
        初始化数据集
        
        Args:
            cfg: 配置对象
            val: 是否为验证集
        """
        self.cfg = cfg
        self.val = val
        self.size = 100 if not val else 20  # 训练集100个样本，验证集20个样本
        
        # 确保cfg是一个对象，有dataset属性
        if hasattr(cfg, 'dataset'):
            dataset_cfg = cfg.dataset
        else:
            dataset_cfg = cfg
            
        # 确保dataset_cfg有相应属性
        self.past_len = getattr(dataset_cfg, 'past_len', 20) if hasattr(dataset_cfg, 'past_len') else dataset_cfg.get('past_len', 20) if isinstance(dataset_cfg, dict) else 20
        self.future_len = getattr(dataset_cfg, 'future_len', 60) if hasattr(dataset_cfg, 'future_len') else dataset_cfg.get('future_len', 60) if isinstance(dataset_cfg, dict) else 60
        
        # 确保cfg.method有相应属性
        if hasattr(cfg, 'method'):
            method_cfg = cfg.method
        else:
            method_cfg = {}
            
        self.max_agents = getattr(method_cfg, 'max_num_agents', 30) if hasattr(method_cfg, 'max_num_agents') else method_cfg.get('max_num_agents', 30) if isinstance(method_cfg, dict) else 30
        self.max_roads = getattr(method_cfg, 'max_num_roads', 100) if hasattr(method_cfg, 'max_num_roads') else method_cfg.get('max_num_roads', 100) if isinstance(method_cfg, dict) else 100
        self.agent_features = getattr(method_cfg, 'num_agent_feature', 32) if hasattr(method_cfg, 'num_agent_feature') else method_cfg.get('num_agent_feature', 32) if isinstance(method_cfg, dict) else 32
        self.road_features = getattr(method_cfg, 'num_map_feature', 64) if hasattr(method_cfg, 'num_map_feature') else method_cfg.get('num_map_feature', 64) if isinstance(method_cfg, dict) else 64
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 生成模拟的输入数据
        input_dict = self._generate_dummy_input(idx)
        return {'input_dict': input_dict}
    
    def _generate_dummy_input(self, idx):
        """生成模拟输入数据"""
        # 生成代理轨迹数据
        obj_trajs = torch.randn(self.max_agents, self.past_len, self.agent_features)
        obj_trajs_mask = torch.ones(self.max_agents, self.past_len)
        # 随机遮蔽一些代理
        mask_indices = torch.rand(self.max_agents) > 0.7
        obj_trajs_mask[mask_indices] = 0
        
        # 生成路网数据
        map_polylines = torch.randn(self.max_roads, 20, self.road_features)  # 假设每条路有20个点
        map_polylines_mask = torch.ones(self.max_roads, 20)
        # 随机遮蔽一些路
        mask_indices = torch.rand(self.max_roads) > 0.5
        map_polylines_mask[mask_indices] = 0
        
        # 生成中心代理索引
        track_index_to_predict = torch.randint(0, self.max_agents, (1,))
        
        # 生成地面真值轨迹
        center_gt_trajs = torch.randn(self.future_len, 2)
        center_gt_trajs_mask = torch.ones(self.future_len)
        center_gt_final_valid_idx = torch.randint(0, self.future_len, (1,))
        
        # 生成其他必要字段
        center_objects_world = torch.randn(7)  # 7维世界坐标
        map_center = torch.randn(2)  # 2维地图中心
        scenario_id = f"scenario_{idx}"
        center_objects_id = f"object_{idx}"
        center_objects_type = torch.randint(1, 4, (1,))  # 1=vehicle, 2=pedestrian, 3=bicycle
        dataset_name = "dummy"
        
        return {
            'obj_trajs': obj_trajs,
            'obj_trajs_mask': obj_trajs_mask,
            'map_polylines': map_polylines,
            'map_polylines_mask': map_polylines_mask,
            'track_index_to_predict': track_index_to_predict,
            'center_gt_trajs': center_gt_trajs,
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_objects_world': center_objects_world,
            'map_center': map_center,
            'scenario_id': scenario_id,
            'center_objects_id': center_objects_id,
            'center_objects_type': center_objects_type,
            'dataset_name': dataset_name
        }
    
    def collate_fn(self, batch):
        """批处理函数"""
        # 合并批次数据
        batched_input_dict = {}
        keys = batch[0]['input_dict'].keys()
        
        for key in keys:
            # 将所有样本的该字段堆叠在一起
            values = [sample['input_dict'][key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                batched_input_dict[key] = torch.stack(values)
            else:
                batched_input_dict[key] = values
        
        return {
            'batch_size': len(batch),
            'input_dict': batched_input_dict
        }