from typing import Optional, Any
import numpy as np
import torch
from .base import BaseReward, RewardResult


class SelfReward(BaseReward):
    def __init__(
        self,
        weight: float = 0.05,
        name: Optional[str] = "self_reward",
        model: Optional[Any] = None,
        prompt_template: str = None,
    ):
        super().__init__(weight=weight, name=name)
        self.model = model
        self.prompt_template = prompt_template or self._default_prompt_template()

    def _default_prompt_template(self) -> str:
        return """请评估以下 SVG 代码重建工程图纸的质量。

图纸描述：{context}

生成的 SVG：
{svg_code}

请从以下几个方面评估：
1. 结构完整性（主结构线是否完整）
2. 线条正确性（线条位置是否准确）
3. 简洁性（是否有冗余线条）

输出格式：score: X.XX（0.0~1.0）

评分："""

    def compute(
        self,
        svg_code: str,
        target: Optional[np.ndarray] = None,
        context: Optional[str] = None,
        **kwargs
    ) -> RewardResult:
        if not self.validate(svg_code):
            return RewardResult(
                total=0.0,
                is_valid=False,
                invalid_penalty=0.5,
                components={"self_reward": 0.0}
            )

        if self.model is None:
            return RewardResult(
                total=0.0,
                is_valid=True,
                components={"self_reward": 0.0, "warning": "No model provided"}
            )

        try:
            score = self._generate_self_score(svg_code, context)
            score = self.normalize_score(score)

            return RewardResult(
                total=self.weight * score,
                is_valid=True,
                components={"self_reward": score}
            )
        except Exception as e:
            return RewardResult(
                total=0.0,
                is_valid=True,
                components={"self_reward": 0.0, "error": str(e)}
            )

    def _generate_self_score(self, svg_code: str, context: Optional[str] = None) -> float:
        if context is None:
            context = "工程图纸，包含各种结构线条"

        prompt = self.prompt_template.format(
            context=context,
            svg_code=svg_code[:500]
        )

        try:
            if hasattr(self.model, "generate"):
                response = self.model.generate(
                    prompt,
                    max_new_tokens=50,
                    do_sample=False,
                )
            elif hasattr(self.model, "__call__"):
                response = self.model(prompt)
            else:
                return 0.5

            score = self._parse_score(response)
            return score

        except Exception:
            return 0.5

    def _parse_score(self, response: str) -> float:
        import re

        patterns = [
            r'score:\s*([0-9.]+)',
            r'评分：\s*([0-9.]+)',
            r'([0-9.]+)\s*/\s*1\.0',
            r'([0-9.]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))
                except ValueError:
                    continue

        return 0.5

    def set_model(self, model: Any):
        self.model = model

    def compute_batch(
        self,
        svg_codes: list,
        contexts: Optional[list] = None,
    ) -> list:
        results = []
        for i, svg in enumerate(svg_codes):
            ctx = contexts[i] if contexts and i < len(contexts) else None
            result = self.compute(svg, context=ctx)
            results.append(result)
        return results
