from typing import Dict, Any, Optional, List, Callable
import json
from pathlib import Path
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from .dataset import BaseDataset, DataSample


class TowerDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        transform: Optional[Callable] = None,
        render_svg: bool = False,
        render_size: tuple = (512, 512),
    ):
        self.render_svg = render_svg
        self.render_size = render_size
        super().__init__(data_dir, split, max_samples, transform)

    def _load_samples(self):
        metadata_path = self.data_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Tower metadata not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            for i, line in enumerate(f):
                if self.max_samples and i >= self.max_samples:
                    break

                data = json.loads(line.strip())
                sample_id = data.get("id", f"tower_{i:04d}")

                image = None
                if "image" in data:
                    img_path = Path(data["image"])
                    if img_path.exists():
                        image = Image.open(img_path).convert("RGB")

                sample = DataSample(
                    id=sample_id,
                    svg=data.get("svg"),
                    image=image,
                    description=data.get("description"),
                    metadata=data.get("metadata"),
                )
                self.samples.append(sample)

    def _prepare_sample(self, sample: DataSample) -> Dict[str, Any]:
        result = {"id": sample.id}

        if sample.svg:
            result["svg"] = sample.svg

            if self.render_svg:
                from ..utils.svg_renderer import render_svg_cairo
                rendered = render_svg_cairo(
                    sample.svg,
                    output_size=self.render_size
                )
                result["rendered"] = torch.from_numpy(rendered).permute(2, 0, 1).float() / 255.0

        if sample.image:
            if self.transform:
                image = self.transform(sample.image)
            else:
                image = sample.image
            result["image"] = image

        if sample.description:
            result["description"] = sample.description

        if sample.metadata:
            result["metadata"] = sample.metadata

        return result

    def split_dataset(self, train_ratio: float = 0.8) -> tuple:
        import random
        random.shuffle(self.samples)

        split_idx = int(len(self.samples) * train_ratio)
        train_samples = self.samples[:split_idx]
        val_samples = self.samples[split_idx:]

        train_dataset = TowerDataset(
            data_dir=self.data_dir,
            split="train",
            max_samples=None,
            transform=self.transform,
            render_svg=self.render_svg,
            render_size=self.render_size,
        )
        train_dataset.samples = train_samples

        val_dataset = TowerDataset(
            data_dir=self.data_dir,
            split="val",
            max_samples=None,
            transform=self.transform,
            render_svg=self.render_svg,
            render_size=self.render_size,
        )
        val_dataset.samples = val_samples

        return train_dataset, val_dataset

    def filter_by_complexity(self, min_lines: int = 0, max_lines: int = 1000) -> "TowerDataset":
        filtered_samples = []
        for sample in self.samples:
            if not sample.metadata:
                continue

            line_count = sample.metadata.get("line_count", 0)
            if min_lines <= line_count <= max_lines:
                filtered_samples.append(sample)

        filtered_dataset = TowerDataset(
            data_dir=self.data_dir,
            split=self.split,
            max_samples=None,
            transform=self.transform,
            render_svg=self.render_svg,
            render_size=self.render_size,
        )
        filtered_dataset.samples = filtered_samples
        return filtered_dataset

    def get_prompt(self, sample_id: str) -> str:
        return f"将这张铁塔工程图纸转换为 SVG 矢量图，保持主材、斜材、横隔等结构。"


class TowerDataAugmentation:
    @staticmethod
    def random_rotate(image: Image.Image, max_angle: float = 15) -> Image.Image:
        import random
        angle = random.uniform(-max_angle, max_angle)
        return image.rotate(angle, fillcolor="white")

    @staticmethod
    def random_brightness(image: Image.Image, factor: float = 0.2) -> Image.Image:
        import random
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        factor = 1.0 + random.uniform(-factor, factor)
        return enhancer.enhance(factor)

    @staticmethod
    def random_contrast(image: Image.Image, factor: float = 0.2) -> Image.Image:
        import random
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        factor = 1.0 + random.uniform(-factor, factor)
        return enhancer.enhance(factor)

    @staticmethod
    def add_noise(image: Image.Image, noise_level: float = 0.01) -> Image.Image:
        import random
        import numpy as np

        img_array = np.array(image).astype(float)
        noise = np.random.randn(*img_array.shape) * noise_level * 255
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    @staticmethod
    def augment(image: Image.Image) -> Image.Image:
        import random
        transforms = [
            TowerDataAugmentation.random_rotate,
            TowerDataAugmentation.random_brightness,
            TowerDataAugmentation.random_contrast,
            TowerDataAugmentation.add_noise,
        ]

        selected = random.sample(transforms, k=random.randint(1, len(transforms)))
        for transform in selected:
            image = transform(image)
        return image
