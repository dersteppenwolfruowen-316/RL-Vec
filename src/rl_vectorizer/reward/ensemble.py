"""集成 Reward 模块。

将多个 Reward 信号（SSIM、CLIP、Keypoint、Complexity、Geometric 等）
按权重加权求和，生成最终奖励值。
"""

from typing import Optional, Dict, Any
import torch
import numpy as np
from .base import RewardResult
from .ssim_reward import SSIMReward
from .clip_reward import CLIPReward
from .keypoint_reward import KeypointReward
from .complexity import ComplexityReward
from .self_reward import SelfReward
from .geometric_reward import GeometricConstraintReward
from .adversarial_reward import AdversarialReward, ConsistencyReward
from ..utils.svg_renderer import render_svg_cairo
from ..utils.svg_validator import validate_svg, count_svg_lines


class EnsembleReward:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        ssim_weight: float = 0.40,
        clip_weight: float = 0.30,
        keypoint_weight: float = 0.20,
        complexity_weight: float = 0.10,
        self_reward_weight: float = 0.00,
        geometric_weight: float = 0.20,
        adversarial_weight: float = 0.10,
        invalid_penalty: float = 0.50,
        device: str = "cuda",
    ):
        if weights is not None:
            self.weights = weights
        else:
            self.weights = {
                "ssim": ssim_weight,
                "clip": clip_weight,
                "keypoint": keypoint_weight,
                "complexity": complexity_weight,
                "self_reward": self_reward_weight,
                "geometric": geometric_weight,
                "adversarial": adversarial_weight,
            }

        self.invalid_penalty = invalid_penalty
        self.device = device

        self.ssim_reward = SSIMReward(weight=self.weights.get("ssim", 0.40))
        self.clip_reward = CLIPReward(weight=self.weights.get("clip", 0.30), device=device)
        self.keypoint_reward = KeypointReward(weight=self.weights.get("keypoint", 0.20))
        self.complexity_reward = ComplexityReward(weight=self.weights.get("complexity", 0.10))
        self.self_reward = SelfReward(weight=0.0)
        self.geometric_reward = GeometricConstraintReward(
            weight=self.weights.get("geometric", 0.20)
        )
        self.adversarial_reward = AdversarialReward(
            weight=self.weights.get("adversarial", 0.10)
        )

        self._validate_weights()

    def _validate_weights(self):
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"Warning: Reward weights sum to {total_weight:.3f}, normalizing to 1.0")
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

        if self.weights.get("self_reward", 0.0) > 0.05:
            print(f"Warning: Self-Reward weight is {self.weights['self_reward']:.3f}, "
                  "consider setting to 0.0 to avoid reward hacking")

    def compute(
        self,
        svg_code: str,
        target_bmp: np.ndarray,
        context: Optional[str] = None,
        target_line_count: Optional[int] = None,
    ) -> RewardResult:
        is_valid = validate_svg(svg_code)

        if not is_valid:
            return RewardResult(
                total=0.0,
                is_valid=False,
                invalid_penalty=self.invalid_penalty,
                components={}
            )

        components = {}

        rendered = render_svg_cairo(svg_code, output_size=(target_bmp.shape[1], target_bmp.shape[0]))

        if self.weights.get("ssim", 0) > 0:
            ssim_result = self.ssim_reward.compute(svg_code, target_bmp)
            components["ssim"] = ssim_result.total

        if self.weights.get("clip", 0) > 0:
            clip_result = self.clip_reward.compute(svg_code, target_bmp)
            components["clip"] = clip_result.total

        if self.weights.get("keypoint", 0) > 0:
            keypoint_result = self.keypoint_reward.compute(svg_code, target_bmp)
            components["keypoint"] = keypoint_result.total

        if self.weights.get("complexity", 0) > 0:
            complexity_result = self.complexity_reward.compute(
                svg_code, target_line_count=target_line_count
            )
            components["complexity"] = complexity_result.total

        if self.weights.get("self_reward", 0) > 0 and context is not None:
            self_reward_result = self.self_reward.compute(svg_code, context=context)
            components["self_reward"] = self_reward_result.total
        else:
            components["self_reward"] = 0.0

        if self.weights.get("geometric", 0) > 0:
            geometric_result = self.geometric_reward.compute(svg_code)
            components["geometric"] = geometric_result.total

        if self.weights.get("adversarial", 0) > 0:
            adversarial_result = self.adversarial_reward.compute(svg_code)
            components["adversarial"] = adversarial_result.total

        total = sum(
            self.weights.get(key, 0) * value
            for key, value in components.items()
        )

        total = max(0.0, min(1.0, total))

        return RewardResult(
            total=total,
            is_valid=True,
            invalid_penalty=0.0,
            components=components
        )

    def compute_detailed(
        self,
        svg_code: str,
        target_bmp: np.ndarray,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self.compute(svg_code, target_bmp, context)

        return {
            "total": result.total,
            "is_valid": result.is_valid,
            "weights": self.weights,
            "components": result.components,
            "penalty": result.invalid_penalty,
        }

    def get_component_names(self) -> list:
        return list(self.weights.keys())

    def set_weight(self, name: str, weight: float):
        if name not in self.weights:
            raise ValueError(f"Unknown reward component: {name}")
        self.weights[name] = weight

    def disable_component(self, name: str):
        if name in self.weights:
            self.weights[name] = 0.0

    def enable_all_components(self):
        default_weights = {
            "ssim": 0.40,
            "clip": 0.30,
            "keypoint": 0.20,
            "complexity": 0.10,
            "self_reward": 0.00,
            "geometric": 0.20,
            "adversarial": 0.10,
        }
        self.weights.update(default_weights)
        self._validate_weights()
