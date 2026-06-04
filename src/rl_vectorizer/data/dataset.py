from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import json
from pathlib import Path
from dataclasses import dataclass
import torch
from ..utils.svg_validator import SVG_NSMAP
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


@dataclass
class DataSample:
    id: str
    svg: Optional[str] = None
    image: Optional[Image.Image] = None
    caption: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_samples = max_samples
        self.transform = transform
        self.samples: List[DataSample] = []

        self._load_samples()

    @abstractmethod
    def _load_samples(self):
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: DataSample) -> Dict[str, Any]:
        result = {"id": sample.id}

        if sample.svg:
            result["svg"] = sample.svg

        if sample.image:
            if self.transform:
                image = self.transform(sample.image)
            else:
                image = sample.image
            result["image"] = image

        if sample.caption:
            result["caption"] = sample.caption

        if sample.description:
            result["description"] = sample.description

        if sample.metadata:
            result["metadata"] = sample.metadata

        return result

    def load_jsonl(self, split: str = "train") -> List[Dict]:
        metadata_path = self.data_dir / f"{split}_metadata.jsonl"
        if not metadata_path.exists():
            metadata_path = self.data_dir / "metadata.jsonl"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        samples = []
        with open(metadata_path, "r") as f:
            for i, line in enumerate(f):
                if self.max_samples and i >= self.max_samples:
                    break
                samples.append(json.loads(line.strip()))
        return samples

    def validate_svg(self, svg_code: str) -> bool:
        from lxml import etree
        try:
            etree.fromstring(svg_code.encode())
            return True
        except Exception:
            return False

    def compute_stats(self) -> Dict[str, Any]:
        from collections import Counter
        from lxml import etree

        line_counts = []
        valid_count = 0
        invalid_count = 0

        for sample in self.samples:
            if sample.svg:
                if self.validate_svg(sample.svg):
                    valid_count += 1
                    try:
                        tree = etree.fromstring(sample.svg.encode())
                        line_count = len(tree.xpath("//svg:line", namespaces=SVG_NSMAP))
                        path_count = len(tree.xpath("//svg:path", namespaces=SVG_NSMAP))
                        line_counts.append(line_count + path_count)
                    except Exception:
                        pass
                else:
                    invalid_count += 1

        return {
            "total": len(self.samples),
            "valid_svg": valid_count,
            "invalid_svg": invalid_count,
            "validity_rate": valid_count / (valid_count + invalid_count) if (valid_count + invalid_count) > 0 else 0,
            "line_count_mean": np.mean(line_counts) if line_counts else 0,
            "line_count_std": np.std(line_counts) if line_counts else 0,
        }
