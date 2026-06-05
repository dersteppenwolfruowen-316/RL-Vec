from typing import Optional
import numpy as np
from .base import BaseReward, RewardResult


class ComplexityReward(BaseReward):
    def __init__(
        self,
        weight: float = 0.10,
        name: Optional[str] = "complexity",
        target_range: tuple = (10, 100),
    ):
        super().__init__(weight=weight, name=name)
        self.target_min, self.target_max = target_range

    def compute(
        self,
        svg_code: str,
        target: Optional[np.ndarray] = None,
        target_line_count: Optional[int] = None,
        **kwargs
    ) -> RewardResult:
        if not self.validate(svg_code):
            return RewardResult(
                total=0.0,
                is_valid=False,
                invalid_penalty=0.5,
                components={"complexity": 0.0}
            )

        try:
            line_count = self._count_svg_elements(svg_code)

            if target_line_count is not None:
                target = target_line_count
            else:
                target = self._estimate_target(target, line_count)

            complexity_score = self._compute_complexity_score(line_count, target)
            complexity_score = self.normalize_score(complexity_score)

            return RewardResult(
                total=self.weight * complexity_score,
                is_valid=True,
                components={"complexity": complexity_score, "line_count": line_count}
            )
        except Exception as e:
            return RewardResult(
                total=0.0,
                is_valid=True,
                components={"complexity": 0.0}
            )

    def _count_svg_elements(self, svg_code: str) -> int:
        line_count = svg_code.count("<line")
        path_count = svg_code.count("<path")
        rect_count = svg_code.count("<rect")
        circle_count = svg_code.count("<circle")
        polyline_count = svg_code.count("<polyline")
        polygon_count = svg_code.count("<polygon")

        return line_count + path_count + rect_count + circle_count + polyline_count + polygon_count

    def _estimate_target(self, img: Optional[np.ndarray], current_count: int) -> int:
        if img is not None:
            height, width = img.shape[:2]
            area = height * width

            estimated_lines = int((area / 10000) * 5)
            return max(self.target_min, min(estimated_lines, self.target_max))

        return int((self.target_min + self.target_max) / 2)

    def _compute_complexity_score(self, line_count: int, target: int) -> float:
        if target == 0:
            return 0.0

        relative_error = abs(line_count - target) / target

        if relative_error <= 0.1:
            score = 1.0 - relative_error * 2.5        
        elif relative_error <= 0.5:
            score = 0.75 - (relative_error - 0.1) * 0.625
        else:
            score = max(0.0, 0.5 - (relative_error - 0.5) * 0.5)

        return float(score)

    def compute_with_penalty(
        self,
        svg_code: str,
        target: Optional[np.ndarray] = None,
        max_lines: int = 500,
    ) -> RewardResult:
        if not self.validate(svg_code):
            return RewardResult(
                total=0.0,
                is_valid=False,
                invalid_penalty=0.5,
                components={"complexity": 0.0}
            )

        line_count = self._count_svg_elements(svg_code)

        if line_count > max_lines:
            penalty = (line_count - max_lines) / max_lines
            score = max(0.0, 0.5 - penalty)

            return RewardResult(
                total=self.weight * score,
                is_valid=True,
                components={"complexity": score, "line_count": line_count, "penalty": penalty}
            )

        return self.compute(svg_code, target)
