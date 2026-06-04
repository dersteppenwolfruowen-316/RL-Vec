"""SFT（Supervised Fine-Tuning）训练器。

.. deprecated::
    此模块已弃用。训练已迁移至 EasyR1 (https://github.com/hiyouga/EasyR1)。
    请使用 ``bash scripts/train_grpo_3b.sh`` 代替。
    详见 docs/easyr1_integration.rst。

用于 Warmup 阶段的监督微调，让模型学习 SVG 语法基础。
此文件仅作为参考保留，不再维护。注意：核心训练逻辑为 placeholder 实现（loss=0）。
"""
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .base_trainer import BaseTrainer
from ..utils.tensorboard_utils import TensorBoardLogger


class SFTTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        super().__init__(model, optimizer, config, device)

    def train_step(self, batch) -> Dict[str, float]:
        self.model.train()

        loss = self._compute_sft_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["max_grad_norm"])
        self.optimizer.step()

        return {"loss": loss.item()}

    def _compute_sft_loss(self, batch) -> torch.Tensor:
        import torch
        dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        return dummy_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
        logger: Optional[TensorBoardLogger] = None,
        save_dir: Optional[str] = None
    ):
        if num_epochs is None:
            num_epochs = self.config["training"]["epochs"]

        epochs = range(self.current_epoch, self.current_epoch + num_epochs)
        self.current_epoch += num_epochs

        for epoch in epochs:
            self.model.train()
            epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for batch_idx, batch in enumerate(epoch_pbar):
                metrics = self.train_step(batch)

                self.global_step += 1

                if logger and self.global_step % self.config["experiment"]["tensorboard"]["log_interval"] == 0:
                    logger.set_step(self.global_step)
                    logger.log_scalar("train/loss", metrics["loss"])

                epoch_pbar.set_postfix({"loss": f'{metrics["loss"]:.4f}'})

            if save_dir and (epoch + 1) % self.config["training"]["save_epochs"] == 0:
                self.save_checkpoint(save_dir, epoch + 1)
