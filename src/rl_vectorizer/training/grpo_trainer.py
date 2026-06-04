"""GRPO（Group Relative Policy Optimization）训练器。

.. deprecated::
    此模块已弃用。训练已迁移至 EasyR1 (https://github.com/hiyouga/EasyR1)。
    请使用 ``bash scripts/train_grpo_3b.sh`` 代替。
    详见 docs/easyr1_integration.rst。

实现 GRPO 强化学习训练流程，包括 group 采样、advantage 估计和策略更新。
此文件仅作为参考保留，不再维护。注意：核心训练逻辑为 placeholder 实现。
"""
import os
import json
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from .base_trainer import BaseTrainer
from ..reward.ensemble import EnsembleReward
from ..utils.tensorboard_utils import TensorBoardLogger
from ..utils.svg_renderer import render_svg_cairo
import numpy as np


class GRPOTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        ref_model,
        reward_fn: EnsembleReward,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        super().__init__(model, optimizer, config, device)
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.group_size = config.get("grpo", {}).get("group_size", 8)
        self.kl_beta = config.get("grpo", {}).get("kl_beta", 0.1)
        self.reward_normalize = config.get("grpo", {}).get("reward_normalize", True)
        self.epsilon = config.get("grpo", {}).get("epsilon", 0.2)
        self.gamma = config.get("grpo", {}).get("gamma", 1.0)
        self.lam = config.get("grpo", {}).get("lambda", 0.95)
        self.entropy_coef = config.get("grpo", {}).get("entropy_coef", 0.01)
        self.clip_advantages = config.get("grpo", {}).get("clip_advantages", True)

        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

    def train_step(
        self,
        batch_bitmaps: torch.Tensor,
        batch_svg_gt: Optional[List[str]] = None,
        batch_context: Optional[List[str]] = None
    ) -> Dict[str, float]:
        self.model.train()
        batch_size = batch_bitmaps.shape[0]

        all_generated_svgs = []
        all_rewards = []
        all_log_probs = []
        all_ref_log_probs = []

        for i in range(batch_size):
            bmp_np = batch_bitmaps[i].cpu().numpy()
            bmp_np = self._preprocess_image(bmp_np)

            context = batch_context[i] if batch_context else None
            svg_gt = batch_svg_gt[i] if batch_svg_gt and i < len(batch_svg_gt) else None

            group_svgs = []
            group_rewards = []
            group_log_probs = []
            group_ref_log_probs = []

            for g in range(self.group_size):
                svg_code, log_prob = self._generate_svg_with_log_prob(bmp_np)
                ref_log_prob = self._get_ref_log_prob(svg_code)

                reward_result = self.reward_fn.compute(svg_code, bmp_np, context)

                group_svgs.append(svg_code)
                group_rewards.append(reward_result.total)
                group_log_probs.append(log_prob)
                group_ref_log_probs.append(ref_log_prob)

            all_generated_svgs.extend(group_svgs)
            all_rewards.extend(group_rewards)
            all_log_probs.extend(group_log_probs)
            all_ref_log_probs.extend(group_ref_log_probs)

        all_rewards_tensor = torch.tensor(all_rewards, device=self.device)
        all_log_probs_tensor = torch.stack(all_log_probs)
        all_ref_log_probs_tensor = torch.stack(all_ref_log_probs)

        rewards_mean = all_rewards_tensor.mean()
        rewards_std = all_rewards_tensor.std()

        if self.reward_normalize and rewards_std > 1e-6:
            normalized_rewards = (all_rewards_tensor - rewards_mean) / rewards_std
        else:
            normalized_rewards = all_rewards_tensor - rewards_mean

        advantages = self._compute_advantages(normalized_rewards, all_log_probs_tensor)

        loss = self._compute_grpo_loss(
            all_log_probs_tensor,
            advantages,
            all_ref_log_probs_tensor
        )

        kl_loss = self._compute_kl_loss(all_log_probs_tensor, all_ref_log_probs_tensor)

        entropy_loss = self._compute_entropy_loss(all_log_probs_tensor)

        total_loss = loss + self.kl_beta * kl_loss - self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config["training"].get("max_grad_norm", 1.0)
        )
        self.optimizer.step()

        with torch.no_grad():
            approx_kl = (all_ref_log_probs_tensor - all_log_probs_tensor).mean().item()

        return {
            "loss": loss.item(),
            "policy_loss": loss.item(),
            "kl_loss": kl_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "mean_reward": rewards_mean.item(),
            "max_reward": all_rewards_tensor.max().item(),
            "min_reward": all_rewards_tensor.min().item(),
            "reward_std": rewards_std.item(),
            "advantage_mean": advantages.mean().item(),
            "approx_kl": approx_kl,
        }

    def _preprocess_image(self, bmp: np.ndarray) -> np.ndarray:
        if bmp.ndim == 3 and bmp.shape[0] == 3:
            bmp = bmp.transpose(1, 2, 0)
        bmp = (bmp * 255).astype("uint8")
        return bmp

    def _generate_svg_with_log_prob(self, bmp: np.ndarray) -> Tuple[str, torch.Tensor]:
        import random
        import re

        prompt = "Convert this engineering drawing to SVG. Preserve all structural lines."

        lines = []
        num_lines = random.randint(5, 15)

        for _ in range(num_lines):
            x1 = random.randint(0, 100)
            y1 = random.randint(0, 100)
            x2 = random.randint(100, 200)
            y2 = random.randint(100, 200)
            lines.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="black" stroke-width="2"/>'
            )

        svg_code = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">'
            f'{"".join(lines)}'
            f'</svg>'
        )

        svg_tokens = re.findall(r'<[^>]+>', svg_code)
        log_prob = torch.tensor(
            [random.uniform(-2, -0.5) for _ in svg_tokens],
            device=self.device
        ).mean()

        return svg_code, log_prob

    def _get_ref_log_prob(self, svg_code: str) -> torch.Tensor:
        if self.ref_model is None:
            return torch.tensor(0.0, device=self.device)

        svg_tokens = len(re.findall(r'<[^>]+>', svg_code))
        ref_log_prob = torch.tensor(
            [random.uniform(-2, -0.5) for _ in range(svg_tokens)],
            device=self.device
        ).mean()

        return ref_log_prob

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor
    ) -> torch.Tensor:
        advantages = rewards - rewards.mean()

        if self.clip_advantages:
            advantages = torch.clamp(advantages, -10, 10)

        return advantages

    def _compute_grpo_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        ref_log_probs: torch.Tensor
    ) -> torch.Tensor:
        ratio = torch.exp(log_probs - ref_log_probs.detach())

        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(
            ratio,
            1.0 - self.epsilon,
            1.0 + self.epsilon
        ) * advantages.detach()

        policy_loss = -torch.min(surr1, surr2).mean()

        return policy_loss

    def _compute_kl_loss(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor
    ) -> torch.Tensor:
        kl_div = F.kl_div(
            log_probs,
            ref_log_probs,
            reduction='batchmean',
            log_target=True
        )

        kl_div = torch.clamp(kl_div, min=0.0, max=10.0)

        return kl_div

    def _compute_entropy_loss(self, log_probs: torch.Tensor) -> torch.Tensor:
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum() / probs.numel()

        return entropy

    def _compute_ref_log_probs_batch(self, svg_codes: List[str]) -> torch.Tensor:
        if self.ref_model is None:
            return torch.zeros(len(svg_codes), device=self.device)

        log_probs = []
        for svg_code in svg_codes:
            svg_tokens = len(re.findall(r'<[^>]+>', svg_code))
            log_prob = torch.tensor(
                [random.uniform(-2, -0.5) for _ in range(svg_tokens)],
                device=self.device
            ).mean()
            log_probs.append(log_prob)

        return torch.stack(log_probs)

    def _generate_dummy_svg(self) -> str:
        import random
        lines = [
            f'<line x1="{random.randint(0, 100)}" y1="{random.randint(0, 100)}" '
            f'x2="{random.randint(100, 200)}" y2="{random.randint(100, 200)}" '
            f'stroke="black" stroke-width="2"/>'
            for _ in range(random.randint(5, 10))
        ]
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">{"".join(lines)}</svg>'

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

        best_reward = float('-inf')

        for epoch in epochs:
            self.model.train()
            epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            epoch_metrics = {
                "loss": [],
                "policy_loss": [],
                "kl_loss": [],
                "entropy_loss": [],
                "mean_reward": [],
                "max_reward": [],
                "min_reward": [],
                "approx_kl": [],
            }

            for batch_idx, batch in enumerate(epoch_pbar):
                batch_bitmaps = batch["image"].to(self.device) if "image" in batch else None
                if batch_bitmaps is None:
                    continue

                batch_svg_gt = batch.get("svg", [None] * len(batch_bitmaps))
                batch_context = batch.get("description", [None] * len(batch_bitmaps))

                metrics = self.train_step(batch_bitmaps, batch_svg_gt, batch_context)

                for key, value in metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key].append(value)

                self.global_step += 1

                if logger and self.global_step % self.config["experiment"]["tensorboard"]["log_interval"] == 0:
                    logger.set_step(self.global_step)
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            logger.log_scalar(f"train/{key}", value)

                epoch_pbar.set_postfix({
                    "loss": f'{metrics["loss"]:.4f}',
                    "reward": f'{metrics["mean_reward"]:.4f}',
                    "kl": f'{metrics["kl_loss"]:.4f}',
                    "approx_kl": f'{metrics.get("approx_kl", 0):.4f}',
                })

            avg_metrics = {key: np.mean(vals) for key, vals in epoch_metrics.items()}

            if logger:
                logger.set_step(self.global_step)
                for key, value in avg_metrics.items():
                    logger.log_scalar(f"epoch/{key}", value)

            if avg_metrics["mean_reward"] > best_reward:
                best_reward = avg_metrics["mean_reward"]
                if save_dir:
                    self.save_checkpoint(save_dir, "best")

            if val_loader is not None and (epoch + 1) % self.config["training"].get("eval_epochs", 1) == 0:
                val_metrics = self.validate(val_loader)
                if logger:
                    for key, value in val_metrics.items():
                        logger.log_scalar(f"val/{key}", value)
                avg_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            if save_dir and (epoch + 1) % self.config["training"].get("save_epochs", 1) == 0:
                self.save_checkpoint(save_dir, epoch + 1)

        return avg_metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_metrics = {
            "reward": [],
            "ssim": [],
            "clip": [],
            "valid_rate": [],
        }

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch_bitmaps = batch["image"].to(self.device)
                batch_svg_gt = batch.get("svg", [None] * len(batch_bitmaps))

                for i in range(len(batch_bitmaps)):
                    bmp_np = self._preprocess_image(batch_bitmaps[i].cpu().numpy())

                    svg_code = self._generate_dummy_svg()
                    reward_result = self.reward_fn.compute(svg_code, bmp_np)

                    all_metrics["reward"].append(reward_result.total)
                    all_metrics["ssim"].append(reward_result.components.get("ssim", 0))
                    all_metrics["clip"].append(reward_result.components.get("clip", 0))
                    all_metrics["valid_rate"].append(1.0 if reward_result.is_valid else 0.0)

        avg_metrics = {key: np.mean(vals) for key, vals in all_metrics.items()}
        return avg_metrics


class GRPOConfig:
    def __init__(
        self,
        group_size: int = 8,
        kl_beta: float = 0.1,
        epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        gamma: float = 1.0,
        lam: float = 0.95,
        reward_normalize: bool = True,
        clip_advantages: bool = True,
        max_grad_norm: float = 1.0,
    ):
        self.group_size = group_size
        self.kl_beta = kl_beta
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.reward_normalize = reward_normalize
        self.clip_advantages = clip_advantages
        self.max_grad_norm = max_grad_norm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_size": self.group_size,
            "kl_beta": self.kl_beta,
            "epsilon": self.epsilon,
            "entropy_coef": self.entropy_coef,
            "gamma": self.gamma,
            "lambda": self.lam,
            "reward_normalize": self.reward_normalize,
            "clip_advantages": self.clip_advantages,
            "max_grad_norm": self.max_grad_norm,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "GRPOConfig":
        return cls(
            group_size=config.get("group_size", 8),
            kl_beta=config.get("kl_beta", 0.1),
            epsilon=config.get("epsilon", 0.2),
            entropy_coef=config.get("entropy_coef", 0.01),
            gamma=config.get("gamma", 1.0),
            lam=config.get("lambda", 0.95),
            reward_normalize=config.get("reward_normalize", True),
            clip_advantages=config.get("clip_advantages", True),
            max_grad_norm=config.get("max_grad_norm", 1.0),
        )


def compute_grpo_advantage(
    rewards: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    normalize: bool = True,
) -> torch.Tensor:
    if baseline is not None:
        advantages = rewards - baseline
    else:
        advantages = rewards - rewards.mean()

    if normalize and rewards.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    advantages = torch.clamp(advantages, -10, 10)

    return advantages


def compute_ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    ratio = torch.exp(log_probs - old_log_probs.detach())

    surr1 = ratio * advantages.detach()
    surr2 = torch.clamp(
        ratio,
        1.0 - epsilon,
        1.0 + epsilon
    ) * advantages.detach()

    loss = -torch.min(surr1, surr2).mean()

    return loss
