"""Reward 基类和结果数据类。

定义 BaseReward 抽象基类和 RewardResult 数据类，所有具体 Reward 均继承自 BaseReward。
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class RewardResult:
    """单次 Reward 计算的结果。"""
    total: float
    is_valid: bool = True
    invalid_penalty: float = 0.0
    components: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.components is None:
            self.components = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "is_valid": self.is_valid,
            "invalid_penalty": self.invalid_penalty,
            "components": self.components,
        }


class BaseReward:
    """所有 Reward 组件的抽象基类。"""

    def __init__(self, weight: float = 1.0, name: Optional[str] = None):
        """初始化 Reward。

        Args:
            weight: 该 Reward 在集成中的权重。
            name: Reward 名称，默认为类名。
        """
        self.weight = weight
        self.name = name or self.__class__.__name__

    def compute(self, svg_code: str, target: np.ndarray, **kwargs) -> RewardResult:
        raise NotImplementedError("Subclasses must implement compute()")

    def validate(self, svg_code: str) -> bool:
        from lxml import etree
        try:
            etree.fromstring(svg_code.encode())
            return True
        except Exception:
            return False

    def render_svg(self, svg_code: str, output_size: tuple = (512, 512)) -> np.ndarray:
        from ..utils.svg_renderer import render_svg_cairo
        return render_svg_cairo(svg_code, output_size=output_size)

    def preprocess_image(self, img: np.ndarray, target_size: Optional[tuple] = None) -> np.ndarray:
        if target_size and img.shape[:2] != target_size:
            from PIL import Image
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize(target_size[::-1], Image.LANCZOS)
            img = np.array(img_pil)
        return img

    def normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        return max(min_val, min(max_val, score))
