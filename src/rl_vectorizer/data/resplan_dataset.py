from typing import Dict, Any, Optional, List, Callable
import json
import os
from pathlib import Path
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from .dataset import BaseDataset, DataSample


class ResPlanDataset(BaseDataset):
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
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            split_dir = self.data_dir

        svg_dir = split_dir / "svg"
        raster_dir = split_dir / "raster"
        annotation_file = split_dir / "annotations.jsonl"

        if not svg_dir.exists():
            raise FileNotFoundError(f"SVG directory not found at {svg_dir}")

        svg_files = list(svg_dir.glob("*.svg"))
        if self.max_samples:
            svg_files = svg_files[:self.max_samples]

        for svg_path in svg_files:
            sample_id = svg_path.stem

            with open(svg_path, "r") as f:
                svg_code = f.read()

            image = None
            if raster_dir.exists():
                raster_path = raster_dir / f"{sample_id}.png"
                if raster_path.exists():
                    image = Image.open(raster_path).convert("RGB")
                else:
                    from ..utils.svg_renderer import render_svg_cairo
                    rendered = render_svg_cairo(svg_code, output_size=self.render_size)
                    image = Image.fromarray(rendered)

            caption = None
            if annotation_file.exists():
                with open(annotation_file, "r") as f:
                    for line in f:
                        data = json.loads(line.strip())
                        if data.get("id") == sample_id:
                            caption = data.get("caption", "")
                            break

            metadata = {
                "source": "resplan",
                "split": self.split,
            }

            sample = DataSample(
                id=sample_id,
                svg=svg_code,
                image=image,
                caption=caption,
                metadata=metadata,
            )
            self.samples.append(sample)

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

        if sample.metadata:
            result["metadata"] = sample.metadata

        return result

    def download_and_prepare(
        output_dir: str,
        url: str = "https://github.com/m-agour/ResPlan/releases/download/v1.0/resplan_v1.tar.gz",
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        archive_path = output_path / "resplan_v1.tar.gz"

        if not archive_path.exists():
            import urllib.request
            print(f"Downloading ResPlan from {url}...")
            urllib.request.urlretrieve(url, archive_path)
            print("Download complete!")

        import tarfile
        print("Extracting...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(output_path)
        print("Extraction complete!")

        archive_path.unlink()

    def get_room_count_distribution(self) -> Dict[str, int]:
        from collections import Counter
        room_counts = Counter()

        for sample in self.samples:
            if sample.metadata and "room_count" in sample.metadata:
                room_counts[sample.metadata["room_count"]] += 1

        return dict(room_counts)

    def filter_by_room_count(self, min_rooms: int = 0, max_rooms: int = 10) -> "ResPlanDataset":
        filtered_samples = []

        for sample in self.samples:
            room_count = 0
            if sample.metadata:
                room_count = sample.metadata.get("room_count", 0)

            if min_rooms <= room_count <= max_rooms:
                filtered_samples.append(sample)

        filtered_dataset = ResPlanDataset(
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
        return "将这张建筑平面图转换为 SVG 矢量图，保持房间布局和墙体结构。"


class ResPlanPreprocessor:
    @staticmethod
    def clean_svg(svg_code: str) -> str:
        import re

        svg_code = re.sub(r'fill="none"', 'fill="#ffffff"', svg_code)
        svg_code = re.sub(r'stroke-width="[^"]*"', 'stroke-width="1"', svg_code)
        svg_code = re.sub(r'<[^>]*opacity="[^"]*"[^>]*>', '', svg_code)

        svg_code = re.sub(r'\s+', ' ', svg_code)
        svg_code = re.sub(r'\s*>\s*<', '><', svg_code)

        return svg_code

    @staticmethod
    def simplify_svg(svg_code: str, tolerance: float = 1.0) -> str:
        import re
        from lxml import etree

        try:
            tree = etree.fromstring(svg_code.encode())
            ns = {"svg": "http://www.w3.org/2000/svg"}

            for path in tree.xpath("//svg:path", namespaces=ns):
                d = path.get("d", "")
                commands = re.findall(r'[MLHVCSQTAZmlhvcsqtaz][^MLHVCSQTAZmlhvcsqtaz]*', d)
                simplified_commands = []
                for cmd in commands:
                    simplified_commands.append(cmd[:20] if len(cmd) > 20 else cmd)
                path.set("d", ''.join(simplified_commands))

            return etree.tostring(tree, encoding='unicode')
        except Exception:
            return svg_code

    @staticmethod
    def extract_structure(svg_code: str) -> Dict[str, Any]:
        from lxml import etree

        try:
            tree = etree.fromstring(svg_code.encode())
            ns = {"svg": "http://www.w3.org/2000/svg"}

            walls = tree.xpath("//svg:line[contains(@stroke, 'black')]", namespaces=ns)
            doors = tree.xpath("//svg:circle", namespaces=ns)
            windows = tree.xpath("//svg:rect[@fill='none']", namespaces=ns)
            rooms = tree.xpath("//svg:text", namespaces=ns)

            return {
                "wall_count": len(walls),
                "door_count": len(doors),
                "window_count": len(windows),
                "room_label_count": len(rooms),
                "total_elements": len(list(tree.iter())),
            }
        except Exception:
            return {}
