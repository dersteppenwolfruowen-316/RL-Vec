from typing import Dict, Any, Optional, Callable, List
import os
import json
from pathlib import Path
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


@dataclass
class SVGStackSample:
    id: str
    svg_code: str
    caption: str
    image: Optional[torch.Tensor] = None


class SVGStackDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_samples: int = 100000,
        render_size: int = 512,
        max_svg_tokens: int = 2048,
        transform: Optional[Callable] = None,
        cache_rendered: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_samples = max_samples
        self.render_size = render_size
        self.max_svg_tokens = max_svg_tokens
        self.transform = transform
        self.cache_rendered = cache_rendered
        self.samples: List[Dict] = []

        self._load_metadata()

    def _load_metadata(self):
        metadata_path = self.data_dir / f"{self.split}_metadata.jsonl"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= self.max_samples:
                        break
                    self.samples.append(json.loads(line.strip()))
        else:
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            "id": sample["id"],
            "svg": sample["svg"],
            "caption": sample.get("caption", ""),
        }

    def get_raw_sample(self, idx: int) -> SVGStackSample:
        sample = self.__getitem__(idx)
        return SVGStackSample(
            id=sample["id"],
            svg_code=sample["svg"],
            caption=sample["caption"],
        )


class SVGStackDownloader:
    def __init__(
        self,
        output_dir: str,
        cache_dir: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.output_dir / ".cache"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_and_prepare(
        self,
        splits: List[str] = ["train", "val"],
        max_samples_per_split: Optional[Dict[str, int]] = None,
    ):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        print(f"Downloading starvector/svg-stack from HuggingFace...")
        ds = load_dataset("starvector/svg-stack", trust_remote_code=True)

        if max_samples_per_split is None:
            max_samples_per_split = {}

        for split in splits:
            if split not in ds:
                print(f"Split '{split}' not found in dataset")
                continue

            split_ds = ds[split]
            max_samples = max_samples_per_split.get(split, len(split_ds))
            split_ds = split_ds.select(range(min(max_samples, len(split_ds))))

            print(f"Processing split '{split}' with {len(split_ds)} samples...")

            metadata_path = self.output_dir / f"{split}_metadata.jsonl"
            with open(metadata_path, "w") as f:
                for i, sample in enumerate(split_ds):
                    if i >= max_samples:
                        break

                    metadata = {
                        "id": sample.get("Filename", f"{split}_{i}"),
                        "svg": sample["Svg"],
                        "caption": sample.get("Captions", ""),
                    }
                    f.write(json.dumps(metadata) + "\n")

            print(f"Saved {i + 1} samples to {metadata_path}")

    def validate_dataset(self, split: str = "train") -> Dict[str, int]:
        metadata_path = self.output_dir / f"{split}_metadata.jsonl"
        if not metadata_path.exists():
            return {"total": 0, "valid": 0, "invalid": 0}

        valid_count = 0
        invalid_count = 0

        from lxml import etree

        with open(metadata_path, "r") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    svg_code = sample.get("svg", "")
                    if svg_code:
                        etree.fromstring(svg_code.encode())
                        valid_count += 1
                    else:
                        invalid_count += 1
                except Exception:
                    invalid_count += 1

        return {
            "total": valid_count + invalid_count,
            "valid": valid_count,
            "invalid": invalid_count,
        }
