"""Reward 系统。

提供多种奖励信号组件和集成奖励，用于 GRPO 强化学习训练。
"""
from .base import BaseReward, RewardResult
from .ssim_reward import SSIMReward
from .clip_reward import CLIPReward
from .keypoint_reward import KeypointReward
from .complexity import ComplexityReward
from .self_reward import SelfReward
from .ensemble import EnsembleReward
from .geometric_reward import GeometricConstraintReward
from .adversarial_reward import AdversarialReward, ConsistencyReward
from .refinement_reward import RefinementReward
from .diffvg_reward import DiffVGVisualReward, CompositeFloorplanReward

__all__ = [
    "BaseReward",
    "RewardResult",
    "SSIMReward",
    "CLIPReward",
    "KeypointReward",
    "ComplexityReward",
    "SelfReward",
    "EnsembleReward",
    "GeometricConstraintReward",
    "AdversarialReward",
    "ConsistencyReward",
    "RefinementReward",
    "DiffVGVisualReward",
    "CompositeFloorplanReward",
]
