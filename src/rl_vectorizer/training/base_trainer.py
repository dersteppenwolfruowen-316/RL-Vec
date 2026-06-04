"""训练器基类。

.. deprecated::
    此模块已弃用。训练已迁移至 EasyR1 (https://github.com/hiyouga/EasyR1)。
    请使用 ``bash scripts/train_grpo_3b.sh`` 代替。
    详见 docs/easyr1_integration.rst。

定义 BaseTrainer 抽象基类，提供训练循环、日志记录和 checkpoint 管理的通用逻辑。
此文件仅作为参考保留，不再维护。
"""
from typing import Dict, Any, Optional
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device

        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float("-inf")

    @abstractmethod
    def train_step(self, batch) -> Dict[str, float]:
        pass

    @abstractmethod
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
        **kwargs
    ):
        pass

    def validation_step(self, batch) -> Dict[str, float]:
        return {}

    def save_checkpoint(
        self,
        save_dir: str,
        step: Optional[int] = None,
        is_best: bool = False,
    ):
        os.makedirs(save_dir, exist_ok=True)

        checkpoint_name = f"checkpoint_{step}" if step else "final_checkpoint"
        checkpoint_path = os.path.join(save_dir, checkpoint_name)

        os.makedirs(checkpoint_path, exist_ok=True)

        model_path = os.path.join(checkpoint_path, "model.pt")
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        config_path = os.path.join(checkpoint_path, "config.yaml")

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)

        import yaml
        with open(config_path, "w") as f:
            yaml.dump(self.config, f)

        metadata = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_metric": self.best_metric,
        }
        metadata_path = os.path.join(checkpoint_path, "metadata.pt")
        torch.save(metadata, metadata_path)

        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        model_path = os.path.join(checkpoint_path, "model.pt")
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        metadata_path = os.path.join(checkpoint_path, "metadata.pt")

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")

        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            print(f"Optimizer loaded from {optimizer_path}")

        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path, map_location=self.device)
            self.global_step = metadata.get("global_step", 0)
            self.current_epoch = metadata.get("current_epoch", 0)
            self.best_metric = metadata.get("best_metric", float("-inf"))
            print(f"Metadata loaded: step={self.global_step}, epoch={self.current_epoch}")

    def update_best_metric(self, metric: float) -> bool:
        if metric > self.best_metric:
            self.best_metric = metric
            return True
        return False

    def clip_gradients(self, max_norm: float = 1.0):
        if self.config.get("training", {}).get("max_grad_norm"):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["training"]["max_grad_norm"]
            )

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step_lr(self, factor: float = 0.1):
        new_lr = self.get_lr() * factor
        self.set_lr(new_lr)
        return new_lr

    def to_device(self, batch):
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch
