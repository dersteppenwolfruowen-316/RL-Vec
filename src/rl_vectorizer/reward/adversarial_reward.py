from typing import Dict, Any, Optional
import numpy as np
from .base import BaseReward, RewardResult


class AdversarialReward(BaseReward):
    def __init__(
        self,
        weight: float = 0.10,
        name: Optional[str] = "adversarial",
        check_duplication: bool = True,
        check_trivial: bool = True,
        check_random: bool = True,
        max_repetition: int = 3,
    ):
        super().__init__(weight=weight, name=name)
        self.check_duplication = check_duplication
        self.check_trivial = check_trivial
        self.check_random = check_random
        self.max_repetition = max_repetition

    def compute(self, svg_code: str, target: Optional[np.ndarray] = None, **kwargs) -> RewardResult:
        if not self.validate(svg_code):
            return RewardResult(
                total=0.0,
                is_valid=False,
                invalid_penalty=0.5,
                components={"adversarial": 0.0}
            )

        penalties = []
        components = {}

        if self.check_duplication:
            dup_penalty, dup_info = self._check_duplication(svg_code)
            penalties.append(dup_penalty)
            components["duplication"] = dup_penalty
            components["duplication_info"] = dup_info

        if self.check_trivial:
            trivial_penalty, trivial_info = self._check_trivial(svg_code)
            penalties.append(trivial_penalty)
            components["trivial"] = trivial_penalty
            components["trivial_info"] = trivial_info

        if self.check_random:
            random_penalty, random_info = self._check_random(svg_code)
            penalties.append(random_penalty)
            components["random"] = random_penalty
            components["random_info"] = random_info

        total_penalty = sum(penalties)
        adversarial_score = max(0.0, 1.0 - total_penalty)

        return RewardResult(
            total=self.weight * adversarial_score,
            is_valid=True,
            components={
                "adversarial": adversarial_score,
                "penalty": total_penalty,
                **components
            }
        )

    def _check_duplication(self, svg_code: str) -> tuple:
        from lxml import etree

        try:
            tree = etree.fromstring(svg_code.encode())
        except:
            return 0.5, "invalid_svg"

        ns = {"svg": "http://www.w3.org/2000/svg"}
        lines = tree.xpath("//svg:line", namespaces=ns)

        if len(lines) < 2:
            return 0.0, "too_few_elements"

        duplicates = 0
        total_pairs = 0

        for i in range(min(len(lines), 10)):
            for j in range(i + 1, min(len(lines), 10)):
                total_pairs += 1

                line1 = lines[i]
                line2 = lines[j]

                x1_1 = float(line1.get("x1", 0))
                y1_1 = float(line1.get("y1", 0))
                x2_1 = float(line1.get("x2", 0))
                y2_1 = float(line1.get("y2", 0))

                x1_2 = float(line2.get("x1", 0))
                y1_2 = float(line2.get("y1", 0))
                x2_2 = float(line2.get("x2", 0))
                y2_2 = float(line2.get("y2", 0))

                if (abs(x1_1 - x1_2) < 1 and abs(y1_1 - y1_2) < 1 and
                    abs(x2_1 - x2_2) < 1 and abs(y2_1 - y2_2) < 1):
                    duplicates += 1

        if total_pairs == 0:
            return 0.0, "no_pairs_checked"

        dup_ratio = duplicates / total_pairs

        if dup_ratio > 0.5:
            penalty = min(1.0, dup_ratio)
        else:
            penalty = 0.0

        return penalty, {"duplicates": duplicates, "total_pairs": total_pairs, "ratio": dup_ratio}

    def _check_trivial(self, svg_code: str) -> tuple:
        from lxml import etree

        try:
            tree = etree.fromstring(svg_code.encode())
        except:
            return 0.5, "invalid_svg"

        ns = {"svg": "http://www.w3.org/2000/svg"}
        lines = tree.xpath("//svg:line", namespaces=ns)

        if len(lines) == 0:
            return 0.5, "no_lines"

        short_lines = 0
        for line in lines:
            x1 = float(line.get("x1", 0))
            y1 = float(line.get("y1", 0))
            x2 = float(line.get("x2", 0))
            y2 = float(line.get("y2", 0))

            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if length < 5:
                short_lines += 1

        short_ratio = short_lines / len(lines)

        if short_ratio > 0.8:
            penalty = min(1.0, short_ratio)
        else:
            penalty = 0.0

        return penalty, {"short_lines": short_lines, "total_lines": len(lines), "ratio": short_ratio}

    def _check_random(self, svg_code: str) -> tuple:
        from lxml import etree

        try:
            tree = etree.fromstring(svg_code.encode())
        except:
            return 0.5, "invalid_svg"

        ns = {"svg": "http://www.w3.org/2000/svg"}
        lines = tree.xpath("//svg:line", namespaces=ns)

        if len(lines) < 3:
            return 0.0, "too_few_elements"

        angles = []
        for line in lines:
            x1 = float(line.get("x1", 0))
            y1 = float(line.get("y1", 0))
            x2 = float(line.get("x2", 0))
            y2 = float(line.get("y2", 0))

            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

        angle_variance = np.var(angles)

        if angle_variance < 10:
            penalty = 0.5
        elif angle_variance > 1000:
            penalty = 0.3
        else:
            penalty = 0.0

        return penalty, {"angle_variance": angle_variance}


class ConsistencyReward(BaseReward):
    def __init__(
        self,
        weight: float = 0.15,
        name: Optional[str] = "consistency",
    ):
        super().__init__(weight=weight, name=name)

    def compute(self, svg_code: str, target: Optional[np.ndarray] = None, **kwargs) -> RewardResult:
        if not self.validate(svg_code):
            return RewardResult(
                total=0.0,
                is_valid=False,
                invalid_penalty=0.5,
                components={"consistency": 0.0}
            )

        try:
            rendered = self.render_svg(svg_code)

            color_variance = np.var(rendered.astype(float))

            if len(rendered.shape) == 3:
                for c in range(rendered.shape[2]):
                    channel_variance = np.var(rendered[:, :, c].astype(float))
                    color_variance = min(color_variance, channel_variance)

            if color_variance < 100:
                consistency_score = 0.3
            elif color_variance > 5000:
                consistency_score = 0.8
            else:
                consistency_score = 0.5

            consistency_score = self.normalize_score(consistency_score)

            return RewardResult(
                total=self.weight * consistency_score,
                is_valid=True,
                components={
                    "consistency": consistency_score,
                    "color_variance": float(color_variance),
                }
            )
        except Exception as e:
            return RewardResult(
                total=0.0,
                is_valid=True,
                components={"consistency": 0.0, "error": str(e)}
            )
