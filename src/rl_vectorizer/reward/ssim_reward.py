from typing import Optional
import numpy as np
from .base import BaseReward, RewardResult


class SSIMReward(BaseReward):
    def __init__(
        self,
        weight: float = 0.40,
        name: Optional[str] = "ssim",
        multichannel: bool = True,
        channel_axis: int = 2,
    ):
        super().__init__(weight=weight, name=name)
        self.multichannel = multichannel
        self.channel_axis = channel_axis

    def compute(self, svg_code: str, target_bmp: np.ndarray, **kwargs) -> RewardResult:
        if not self.validate(svg_code):
            return RewardResult(
                total=0.0,
                is_valid=False,
                invalid_penalty=0.5,
                components={"ssim": 0.0}
            )

        try:
            rendered = self.render_svg(svg_code, output_size=(target_bmp.shape[1], target_bmp.shape[0]))

            if rendered.shape != target_bmp.shape:
                rendered = self.preprocess_image(rendered, target_bmp.shape[:2])

            ssim_value = self._compute_ssim(rendered, target_bmp)
            ssim_value = self.normalize_score(ssim_value)

            return RewardResult(
                total=self.weight * ssim_value,
                is_valid=True,
                components={"ssim": ssim_value}
            )
        except Exception as e:
            return RewardResult(
                total=0.0,
                is_valid=True,
                components={"ssim": 0.0}
            )

    def _compute_ssim(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        data_range: float = 255.0,
    ) -> float:
        from skimage.metrics import structural_similarity as ssim

        if img1.shape != img2.shape:
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]

        if len(img1.shape) == 3 and self.multichannel:
            score = ssim(img1, img2, channel_axis=self.channel_axis, data_range=data_range)
        else:
            score = ssim(img1, img2, data_range=data_range)

        return float(score)

    def compute_multiscale(
        self,
        svg_code: str,
        target_bmp: np.ndarray,
        scales: list = [1.0, 0.5, 0.25],
    ) -> RewardResult:
        if not self.validate(svg_code):
            return RewardResult(
                total=0.0,
                is_valid=False,
                invalid_penalty=0.5,
                components={"ssim": 0.0}
            )

        rendered = self.render_svg(svg_code, output_size=(target_bmp.shape[1], target_bmp.shape[0]))
        scores = []

        for scale in scales:
            h, w = int(target_bmp.shape[0] * scale), int(target_bmp.shape[1] * scale)

            from PIL import Image
            img1_scaled = np.array(Image.fromarray(rendered).resize((w, h), Image.LANCZOS))
            img2_scaled = np.array(Image.fromarray(target_bmp).resize((w, h), Image.LANCZOS))

            score = self._compute_ssim(img1_scaled, img2_scaled)
            scores.append(score)

        avg_score = np.mean(scores)
        avg_score = self.normalize_score(avg_score)

        return RewardResult(
            total=self.weight * avg_score,
            is_valid=True,
            components={"ssim": avg_score, "ssim_multiscale": scores}
        )
