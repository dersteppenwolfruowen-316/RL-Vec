"""DiffRasterizer Refinement Reward — 优化前后 loss 差值为奖励。"""
from typing import Optional
import torch
import numpy as np
from .base import BaseReward, RewardResult


class RefinementReward(BaseReward):
    """基于 DiffRasterizer 优化增量奖励。"""

    def __init__(
        self,
        weight: float = 0.30,
        name: Optional[str] = "refinement",
        num_steps: int = 50,
        lr: float = 20.0,
        stroke_width: float = 2.0,
        H: int = 256,
        W: int = 256,
    ):
        super().__init__(weight=weight, name=name)
        self.num_steps = num_steps
        self.lr = lr
        self.stroke_width = stroke_width
        self.H = H
        self.W = W

    def compute(self, svg_code: str, target_bmp: np.ndarray, **kwargs) -> RewardResult:
        if not self.validate(svg_code):
            return RewardResult(total=0.0, is_valid=False,
                                invalid_penalty=0.5,
                                components={"refinement": 0.0})

        try:
            # 从 SVG 提取线段
            from ..utils.diff_line_rasterizer import (
                DifferentiableLineRasterizer,
            )
            rasterizer = DifferentiableLineRasterizer(
                H=self.H, W=self.W,
                stroke_width=self.stroke_width,
                mode="gaussian",
            )

            lines = rasterizer._extract_lines_from_svg(svg_code)
            if len(lines) == 0:
                return RewardResult(total=0.0, is_valid=True,
                                    components={"refinement": 0.0,
                                                "reason": "no_lines"})

            # 缩放到渲染尺寸
            scale_x = self.W / target_bmp.shape[1]
            scale_y = self.H / target_bmp.shape[0]
            lines_scaled = lines * torch.tensor([scale_x, scale_y, scale_x, scale_y])
            target_tensor = torch.from_numpy(target_bmp).float() / 255.0
            target_resized = torch.nn.functional.interpolate(
                target_tensor.permute(2, 0, 1).unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
            )[0].permute(1, 2, 0)

            # 优化前 loss
            with torch.no_grad():
                before_img = rasterizer(lines_scaled)
            before_loss = ((before_img - target_resized) ** 2).mean().item()

            if before_loss < 0.001:
                # 已经几乎完美
                return RewardResult(
                    total=self.weight * 1.0, is_valid=True,
                    components={"refinement": 1.0, "improvement": 0.0}
                )

            # DiffRasterizer 优化
            opt_lines = lines_scaled.clone().detach().requires_grad_(True)
            optimizer = torch.optim.SGD([opt_lines], lr=self.lr, momentum=0.9)
            sigmas = torch.linspace(
                self.stroke_width * 2, self.stroke_width * 0.5, self.num_steps
            )

            for step in range(self.num_steps):
                optimizer.zero_grad()
                sigma = sigmas[step].item()
                r = DifferentiableLineRasterizer(
                    H=self.H, W=self.W,
                    stroke_width=sigma, mode="gaussian",
                )
                img = r(opt_lines, stroke_widths=torch.tensor([sigma]))
                diff = (img - target_resized).abs()
                k = max(1, int(0.3 * diff.numel()))
                loss = diff.view(-1).topk(k)[0].mean()
                loss.backward()
                optimizer.step()

            # 优化后 loss
            with torch.no_grad():
                after_rasterizer = DifferentiableLineRasterizer(
                    H=self.H, W=self.W,
                    stroke_width=self.stroke_width,
                    mode="gaussian",
                )
                after_img = after_rasterizer(opt_lines.detach())
            after_loss = ((after_img - target_resized) ** 2).mean().item()

            # Reward = 提升量
            improvement = max(0.0, before_loss - after_loss)
            # 归一化到 [0, 1]
            max_improvement = 0.1
            score = min(1.0, improvement / max_improvement)

            return RewardResult(
                total=self.weight * score,
                is_valid=True,
                components={
                    "refinement": score,
                    "before_loss": before_loss,
                    "after_loss": after_loss,
                    "improvement": improvement,
                }
            )

        except Exception as e:
            return RewardResult(
                total=0.0, is_valid=True,
                components={"refinement": 0.0, "error": str(e)}
            )
