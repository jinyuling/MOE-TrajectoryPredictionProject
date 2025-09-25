import pytorch_lightning as pl
import torch
import wandb
torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from dummy_dataset import DummyDataset  # 使用模拟数据集
from utils.utils import set_seed, find_latest_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import os
import csv

# 添加一个简单的build_dataset函数
def build_dataset(cfg, val=False):
    return DummyDataset(cfg, val)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    
    # 简化配置处理
    model_cfg = cfg

    model = build_model(model_cfg)

    # ✅ 加载专家预训练权重（冻结专家，只训练Router）
    # 简化处理，跳过专家权重加载
    if hasattr(model_cfg.method, 'model_name') and model_cfg.method.model_name == 'MOE':
        # ✅ 冻结专家参数，只训练 Router
        for expert in model.experts:
            for param in expert.parameters():
                param.requires_grad = False
        print("🔒 All expert parameters frozen, only Router will be trained")

    train_set = build_dataset(model_cfg)
    val_set = build_dataset(model_cfg, val=True)

    train_batch_size = max(model_cfg.method['train_batch_size'] // len(model_cfg.devices),  1)
    eval_batch_size = max(model_cfg.method['eval_batch_size'] // len(model_cfg.devices), 1)

    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        monitor='val/brier_fde',  # Replace with your validation metric
        filename='{epoch}-{val/brier_fde:.2f}-{val/minFDE6:.2f}-{val/minADE6:.2f}',
        save_top_k=1,
        mode='min',  # 'min' for loss/error, 'max' for accuracy
        dirpath=f'./unitraj_ckpt/{model_cfg.exp_name}'
    )

    call_backs.append(checkpoint_callback)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=model_cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=model_cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=train_set.collate_fn)

    trainer = pl.Trainer(
        max_epochs=model_cfg.method.max_epochs,
        logger=None if model_cfg.debug else WandbLogger(project="unitraj", name=model_cfg.exp_name, id=model_cfg.exp_name),
        devices=1 if model_cfg.debug else model_cfg.devices,
        gradient_clip_val=model_cfg.method.grad_clip_norm,
        # accumulate_grad_batches=model_cfg.method.Trainer.accumulate_grad_batches,
        accelerator="cpu" if model_cfg.debug else "gpu",
        profiler="simple",
        strategy="auto" if model_cfg.debug else "auto", # 
        callbacks=call_backs
    )


    if model_cfg.ckpt_path is None and not model_cfg.debug:
        # Pattern to match all .ckpt files in the base_path recursively
        search_pattern = os.path.join('/home/zzs/zzs/', model_cfg.exp_name, '**', '*.ckpt')
        model_cfg.ckpt_path = find_latest_checkpoint(search_pattern)

    # trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=model_cfg.ckpt_path,ckpt_path_autobot=model_cfg.ckpt_path_autobot,ckpt_path_wayformer=model_cfg.ckpt_path_wayformer)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=model_cfg.ckpt_path)

    
if __name__ == '__main__':
    wandb.init(mode="offline")
    train()