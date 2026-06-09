"""DiffVG 可微渲染视觉奖励。

将模型生成的 SVG 通过 DiffVG 渲染为光栅图（支持自动求导），
与目标位图计算像素级损失作为视觉 reward。

在 DiffVG 不可用时自动回退到 cairosvg 渲染（无梯度，仅推理评估）。
"""
from typing import Optional, Dict, Any, Tuple
import torch
import numpy as np
from .base import BaseReward, RewardResult


class DiffVGVisualReward(BaseReward):
    """基于 DiffVG 渲染的视觉相似性奖励。

    将 SVG 渲染为光栅图，计算与目标位图的 SSIM 和 L1 相似度。
    支持可微渲染（训练时）和快速回退（评估时）。
    """

    def __init__(
        self,
        weight: float = 0.35,
        name: Optional[str] = "diffvg_visual",
        render_size: Tuple[int, int] = (112, 112),
        ssim_weight: float = 0.6,
        l1_weight: float = 0.4,
        device: str = "cuda",
    ):
        super().__init__(weight=weight, name=name)
        self.render_size = render_size
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.device = device
        self._renderer = None

    def _get_renderer(self):
        if self._renderer is None:
            try:
                from ..rl.diffvg_renderer import DiffVGRenderer
                self._renderer = DiffVGRenderer(self.device)
            except ImportError:
                self._renderer = None
        return self._renderer

    def render_to_tensor(self, svg_code: str) -> Optional[torch.Tensor]:
        """将 SVG 渲染为 [1, 3, H, W] 的 torch tensor。

        优先使用 DiffVG（可微），回退到 cairosvg（不可微）。
        """
        renderer = self._get_renderer()
        if renderer is None:
            return None
        try:
            w, h = self.render_size
            rendered = renderer.render(svg_code, width=w, height=h)
            # rendered shape: [1, 3, H, W] for DiffVG
            # or [1, H, W, 3] for cairosvg fallback — normalize
            if rendered.shape[1] != 3:
                rendered = rendered.permute(0, 3, 1, 2)
            return rendered
        except Exception:
            return None

    def _compute_ssim_torch(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """简化的 batched SSIM 计算，纯 PyTorch 实现（无 skimage 依赖）。"""
        # 使用简单的局部统计量近似 SSIM
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Gaussian 模糊核 (size=5, sigma=1.5)
        kernel = self._gaussian_kernel(5, 1.5).to(pred.device)
        kernel = kernel.expand(3, 1, 5, 5)  # [C, 1, 5, 5]

        # 计算均值
        mu_pred = torch.nn.functional.conv2d(pred, kernel, groups=3, padding=2)
        mu_target = torch.nn.functional.conv2d(target, kernel, groups=3, padding=2)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        # 计算方差
        sigma_pred_sq = torch.nn.functional.conv2d(
            pred ** 2, kernel, groups=3, padding=2
        ) - mu_pred_sq
        sigma_target_sq = torch.nn.functional.conv2d(
            target ** 2, kernel, groups=3, padding=2
        ) - mu_target_sq
        sigma_pred_target = torch.nn.functional.conv2d(
            pred * target, kernel, groups=3, padding=2
        ) - mu_pred_target

        # SSIM 公式
        numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
        denominator = (mu_pred_sq + mu_target_sq + C1) * (
            sigma_pred_sq + sigma_target_sq + C2
        )
        ssim_map = numerator / denominator
        return ssim_map.mean()

    def _gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """生成 1D Gaussian 核，用于构造 2D 核。"""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g[:, None] * g[None, :]

    def compute(
        self,
        svg_code: str,
        target: np.ndarray,
        **kwargs,
    ) -> RewardResult:
        if not self.validate(svg_code):
            return RewardResult(
                total=0.0,
                is_valid=False,
                invalid_penalty=0.5,
                components={"diffvg_ssim": 0.0, "diffvg_l1": 0.0},
            )

        # 将 target numpy 转为 tensor
        if isinstance(target, np.ndarray):
            if target.dtype == np.uint8:
                target_float = target.astype(np.float32) / 255.0
            else:
                target_float = target.astype(np.float32)
            target_t = torch.from_numpy(target_float).to(self.device)
            # 转为 [1, 3, H, W]
            if target_t.dim() == 3:
                if target_t.shape[-1] == 3:  # HWC
                    target_t = target_t.permute(2, 0, 1)
                target_t = target_t.unsqueeze(0)
            elif target_t.dim() == 2:
                target_t = target_t.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1)
        else:
            target_t = target

        # resize target 到 render_size
        if target_t.shape[2:] != self.render_size:
            target_t = torch.nn.functional.interpolate(
                target_t, size=self.render_size, mode="bilinear", align_corners=False
            )

        rendered = self.render_to_tensor(svg_code)
        if rendered is None:
            return RewardResult(
                total=0.0,
                is_valid=False,
                components={"diffvg_ssim": 0.0, "diffvg_l1": 0.0},
            )

        # 对齐尺寸
        if rendered.shape[2:] != target_t.shape[2:]:
            rendered = torch.nn.functional.interpolate(
                rendered, size=target_t.shape[2:], mode="bilinear", align_corners=False
            )

        # 计算 SSIM（纯 PyTorch）
        with torch.no_grad():
            ssim_val = self._compute_ssim_torch(rendered, target_t).item()
            ssim_val = max(0.0, min(1.0, ssim_val))

        # 计算 L1 相似度 (1 - L1 loss)
        l1_loss = torch.abs(rendered - target_t).mean().item()
        l1_sim = max(0.0, 1.0 - l1_loss)

        # 组合
        visual_score = self.ssim_weight * ssim_val + self.l1_weight * l1_sim
        visual_score = self.normalize_score(visual_score)

        return RewardResult(
            total=self.weight * visual_score,
            is_valid=True,
            components={
                "diffvg_ssim": ssim_val,
                "diffvg_l1": l1_sim,
                "visual": visual_score,
            },
        )


class CompositeFloorplanReward(BaseReward):
    """组合奖励：整合多个 reward 组件用于建筑平面图生成。

    包括：
    - R_validity: SVG 合法性和完整性
    - R_geometry: 几何约束（外墙封闭、房间不重叠、门在墙上等）
    - R_visual: DiffVG 渲染后的视觉相似性

    权重随训练阶段动态调整。
    """

    def __init__(
        self,
        weight: float = 1.0,
        name: Optional[str] = "composite_floorplan",
        validity_weight: float = 0.30,
        geometry_weight: float = 0.25,
        visual_weight: float = 0.25,
        intermediate_weight: float = 0.20,
        device: str = "cuda",
        render_size: Tuple[int, int] = (112, 112),
    ):
        super().__init__(weight=weight, name=name)
        self.validity_weight = validity_weight
        self.geometry_weight = geometry_weight
        self.visual_weight = visual_weight
        self.intermediate_weight = intermediate_weight
        self.device = device

        # 子 reward 组件（惰性初始化）
        self._visual_reward = None
        self._geometry_reward = None
        self._render_size = render_size

    def _get_visual_reward(self):
        if self._visual_reward is None:
            self._visual_reward = DiffVGVisualReward(
                weight=1.0,  # 权重由 Composite 统一管理
                render_size=self._render_size,
                device=self.device,
            )
        return self._visual_reward

    def _get_geometry_reward(self):
        if self._geometry_reward is None:
            try:
                from .geometric_reward import GeometricConstraintReward
                self._geometry_reward = GeometricConstraintReward(weight=1.0)
            except ImportError:
                self._geometry_reward = None
        return self._geometry_reward

    def compute(
        self,
        svg_code: str,
        target: np.ndarray,
        intermediate_xml: Optional[str] = None,
        **kwargs,
    ) -> RewardResult:
        components = {}

        # 1) R_validity — SVG 能否解析
        is_valid = self.validate(svg_code)
        validity_score = 1.0 if is_valid else 0.0
        components["validity"] = validity_score

        if not is_valid:
            return RewardResult(
                total=0.0,
                is_valid=False,
                invalid_penalty=0.5,
                components=components,
            )

        # 2) R_geometry — 几何约束
        geometry_reward = self._get_geometry_reward()
        if geometry_reward and target is not None:
            try:
                geo_result = geometry_reward.compute(svg_code, target)
                geom_score = geo_result.total / max(geo_result.weight, 1e-6)
                components["geometry"] = geom_score
                # 传递子组件
                for k, v in geo_result.components.items():
                    components[f"geo_{k}"] = v
            except Exception:
                geom_score = 0.5
                components["geometry"] = geom_score
        else:
            geom_score = 0.5
            components["geometry"] = geom_score

        # 3) R_visual — DiffVG 渲染视觉相似性
        visual_reward = self._get_visual_reward()
        if visual_reward and target is not None:
            try:
                vis_result = visual_reward.compute(svg_code, target)
                vis_score = vis_result.total / max(vis_result.weight, 1e-6)
                for k, v in vis_result.components.items():
                    components[f"vis_{k}"] = v
            except Exception:
                vis_score = 0.0
                components["visual"] = vis_score
        else:
            vis_score = 0.0
            components["visual"] = vis_score

        # 4) R_intermediate — 中间指令质量（如有）
        intermediate_score = 0.5  # 默认中性
        if intermediate_xml:
            try:
                intermediate_score = self._score_intermediate(intermediate_xml)
            except Exception:
                intermediate_score = 0.5
        components["intermediate"] = intermediate_score

        # 加权组合
        total_score = (
            self.validity_weight * validity_score
            + self.geometry_weight * geom_score
            + self.visual_weight * vis_score
            + self.intermediate_weight * intermediate_score
        )
        total_score = self.normalize_score(total_score)

        return RewardResult(
            total=total_score,
            is_valid=True,
            components=components,
        )

    def _score_intermediate(self, xml_str: str) -> float:
        """评估中间指令格式质量（简单规则）。"""
        import re

        score = 0.0
        checks = 0

        required_tags = [
            r"<analysis>", r"</analysis>",
            r"<outer_wall>", r"</outer_wall>",
        ]
        for tag in required_tags:
            if tag in xml_str:
                score += 1.0
            checks += 1

        # 检查 polygon 坐标格式
        poly_matches = re.findall(r"polygon=\(.*?\)", xml_str)
        for pm in poly_matches:
            try:
                coords = pm.replace("polygon=(", "").replace(")", "")
                parts = coords.split(",")
                # 成对坐标，至少 3 点才能构成多边形
                if len(parts) >= 6 and len(parts) % 2 == 0:
                    score += 1.0
            except Exception:
                pass
            checks += 1

        # 检查 svg_output 存在
        if "<svg_output>" in xml_str and "</svg_output>" in xml_str:
            score += 1.0
        checks += 1

        return max(0.0, min(1.0, score / max(checks, 1)))
