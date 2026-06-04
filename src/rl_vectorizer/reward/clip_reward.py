from typing import Optional, Union
import numpy as np
import torch
from PIL import Image
from .base import BaseReward, RewardResult


class CLIPReward(BaseReward):
    def __init__(
        self,
        weight: float = 0.30,
        name: Optional[str] = "clip",
        model_name: str = "ViT-B/32",
        pretrained: str = "openai",
        device: Optional[str] = None,
    ):
        super().__init__(weight=weight, name=name)
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._preprocess = None

    @property
    def model(self):
        if self._model is None:
            import open_clip
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained
            )
            self._model = self._model.to(self.device)
            self._model.eval()
        return self._model

    @property
    def preprocess(self):
        if self._preprocess is None:
            _ = self.model
        return self._preprocess

    def compute(self, svg_code: str, target_bmp: np.ndarray, **kwargs) -> RewardResult:
        if not self.validate(svg_code):
            return RewardResult(
                total=0.0,
                is_valid=False,
                invalid_penalty=0.5,
                components={"clip": 0.0}
            )

        try:
            rendered = self.render_svg(svg_code, output_size=(target_bmp.shape[1], target_bmp.shape[0]))

            clip_score = self._compute_clip_score(rendered, target_bmp)
            clip_score = self.normalize_score(clip_score)

            return RewardResult(
                total=self.weight * clip_score,
                is_valid=True,
                components={"clip": clip_score}
            )
        except Exception as e:
            return RewardResult(
                total=0.0,
                is_valid=True,
                components={"clip": 0.0}
            )

    def _compute_clip_score(self, img1: np.ndarray, img2: np.ndarray) -> float:
        with torch.no_grad():
            if isinstance(img1, np.ndarray):
                img1 = Image.fromarray(img1)
            if isinstance(img2, np.ndarray):
                img2 = Image.fromarray(img2)

            img1_tensor = self.preprocess(img1).unsqueeze(0).to(self.device)
            img2_tensor = self.preprocess(img2).unsqueeze(0).to(self.device)

            feat1 = self.model.encode_image(img1_tensor)
            feat2 = self.model.encode_image(img2_tensor)

            feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
            feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)

            similarity = (feat1 * feat2).sum(dim=-1)

            return float(similarity.cpu())

    def compute_text_image_similarity(
        self,
        svg_code: str,
        target_bmp: np.ndarray,
        text: str,
    ) -> RewardResult:
        if not self.validate(svg_code):
            return RewardResult(
                total=0.0,
                is_valid=False,
                invalid_penalty=0.5,
                components={"clip_text": 0.0}
            )

        try:
            rendered = self.render_svg(svg_code, output_size=(target_bmp.shape[1], target_bmp.shape[0]))

            with torch.no_grad():
                if isinstance(rendered, np.ndarray):
                    rendered = Image.fromarray(rendered)

                img_tensor = self.preprocess(rendered).unsqueeze(0).to(self.device)
                text_tokens = open_clip.tokenize([text]).to(self.device)

                img_feat = self.model.encode_image(img_tensor)
                text_feat = self.model.encode_text(text_tokens)

                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

                similarity = (img_feat * text_feat).sum(dim=-1)
                score = float(similarity.cpu())

            score = self.normalize_score(score)

            return RewardResult(
                total=self.weight * score,
                is_valid=True,
                components={"clip_text": score}
            )
        except Exception:
            return RewardResult(
                total=0.0,
                is_valid=True,
                components={"clip_text": 0.0}
            )
