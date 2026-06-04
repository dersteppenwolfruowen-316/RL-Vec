from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
import random
import json
from pathlib import Path


class TowerSVGGenerator:
    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        min_height: float = 0.5,
        max_height: float = 0.9,
    ):
        self.width = width
        self.height = height
        self.min_height = min_height
        self.max_height = max_height

    def generate(
        self,
        num_main_members: int = 10,
        num_secondary_members: int = 20,
        num_diaphragms: int = 3,
        slope_angle: float = 75.0,
    ) -> Tuple[str, Dict[str, Any]]:
        svg_lines = []
        metadata = {
            "num_main_members": num_main_members,
            "num_secondary_members": num_secondary_members,
            "num_diaphragms": num_diaphragms,
            "slope_angle": slope_angle,
        }

        svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}">')

        svg_lines.append(f'<!-- Main Members -->')
        main_lines = self._generate_main_members(num_main_members, slope_angle)
        svg_lines.extend(main_lines)
        metadata["main_lines"] = main_lines

        svg_lines.append(f'<!-- Secondary Members -->')
        secondary_lines = self._generate_secondary_members(main_lines)
        svg_lines.extend(secondary_lines)

        svg_lines.append(f'<!-- Diaphragms -->')
        diaphragm_lines = self._generate_diaphragms(num_diaphragms)
        svg_lines.extend(diaphragm_lines)

        svg_lines.append('</svg>')

        svg_code = '\n'.join(svg_lines)

        return svg_code, metadata

    def _generate_main_members(self, num_members: int, slope_angle: float) -> List[str]:
        lines = []

        slope_rad = np.radians(slope_angle)
        center_x = self.width / 2
        base_y = self.height * 0.9
        top_y = self.height * random.uniform(self.min_height, self.max_height)

        top_width = self.width * 0.3

        for i in range(num_members):
            x_offset = (i - num_members / 2) * (top_width / num_members)
            x = center_x + x_offset

            lines.append(
                f'<line x1="{center_x}" y1="{base_y}" x2="{x}" y2="{top_y}" '
                f'stroke="black" stroke-width="3"/>'
            )

        base_left = center_x - top_width / 2
        base_right = center_x + top_width / 2

        lines.append(
            f'<line x1="{base_left}" y1="{base_y}" x2="{center_x - top_width / 4}" y2="{top_y}" '
            f'stroke="black" stroke-width="3"/>'
        )
        lines.append(
            f'<line x1="{base_right}" y1="{base_y}" x2="{center_x + top_width / 4}" y2="{top_y}" '
            f'stroke="black" stroke-width="3"/>'
        )

        return lines

    def _generate_secondary_members(self, main_lines: List[str]) -> List[str]:
        lines = []

        num_secondary = random.randint(15, 25)

        for _ in range(num_secondary):
            x1 = random.randint(int(self.width * 0.2), int(self.width * 0.8))
            y1 = random.randint(int(self.height * 0.3), int(self.height * 0.85))
            x2 = random.randint(int(self.width * 0.2), int(self.width * 0.8))
            y2 = random.randint(int(self.height * 0.3), int(self.height * 0.85))

            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                lines.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                    f'stroke="black" stroke-width="1.5"/>'
                )

        return lines

    def _generate_diaphragms(self, num_diaphragms: int) -> List[str]:
        lines = []

        center_x = self.width / 2
        base_y = self.height * 0.9
        top_y = self.height * self.min_height

        diaphragm_ys = np.linspace(top_y, base_y, num_diaphragms + 2)[1:-1]

        top_width_at_base = self.width * 0.3
        top_width_at_top = self.width * 0.05

        for i, y in enumerate(diaphragm_ys):
            progress = i / (len(diaphragm_ys) - 1) if len(diaphragm_ys) > 1 else 0
            width = top_width_at_base - (top_width_at_base - top_width_at_top) * progress

            left_x = center_x - width / 2
            right_x = center_x + width / 2

            lines.append(
                f'<line x1="{left_x}" y1="{y}" x2="{right_x}" y2="{y}" '
                f'stroke="black" stroke-width="2"/>'
            )

        return lines


class SyntheticTowerDataset:
    def __init__(
        self,
        output_dir: str,
        num_samples: int = 100,
        width: int = 512,
        height: int = 512,
    ):
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.generator = TowerSVGGenerator(width=width, height=height)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self):
        samples = []

        for i in range(self.num_samples):
            num_main = random.randint(6, 14)
            num_secondary = random.randint(15, 30)
            num_diaphragms = random.randint(2, 5)
            slope_angle = random.uniform(70, 85)

            svg_code, metadata = self.generator.generate(
                num_main_members=num_main,
                num_secondary_members=num_secondary,
                num_diaphragms=num_diaphragms,
                slope_angle=slope_angle,
            )

            sample = {
                "id": f"synthetic_tower_{i:04d}",
                "svg": svg_code,
                "metadata": metadata,
                "source": "synthetic",
            }

            samples.append(sample)

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{self.num_samples} samples")

        metadata_path = self.output_dir / "metadata.jsonl"
        with open(metadata_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        print(f"\nSaved {len(samples)} samples to {metadata_path}")

        return samples

    def generate_with_images(self):
        samples = self.generate()

        svg_dir = self.output_dir / "svg"
        raster_dir = self.output_dir / "raster"
        svg_dir.mkdir(exist_ok=True)
        raster_dir.mkdir(exist_ok=True)

        from ..utils.svg_renderer import render_svg_cairo

        for sample in samples:
            sample_id = sample["id"]

            svg_path = svg_dir / f"{sample_id}.svg"
            with open(svg_path, "w") as f:
                f.write(sample["svg"])

            rendered = render_svg_cairo(sample["svg"], output_size=(self.generator.width, self.generator.height))
            raster_path = raster_dir / f"{sample_id}.png"
            Image.fromarray(rendered).save(raster_path)

            sample["svg_path"] = str(svg_path)
            sample["image"] = str(raster_path)

        metadata_path = self.output_dir / "metadata.jsonl"
        with open(metadata_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        return samples


def generate_floor_plan(
    width: int = 512,
    height: int = 512,
    num_rooms: int = 4,
) -> str:
    svg_lines = []

    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')

    wall_width = 5

    svg_lines.append(f'<!-- Outer Walls -->')
    svg_lines.append(
        f'<rect x="20" y="20" width="{width - 40}" height="{height - 40}" '
        f'fill="none" stroke="black" stroke-width="{wall_width}"/>'
    )

    num_horizontal = random.randint(1, 3)
    num_vertical = random.randint(1, 3)

    svg_lines.append(f'<!-- Interior Walls -->')

    for i in range(num_horizontal):
        y = int(height * (i + 1) / (num_horizontal + 1))
        svg_lines.append(
            f'<line x1="20" y1="{y}" x2="{width - 20}" y2="{y}" '
            f'stroke="black" stroke-width="{wall_width - 1}"/>'
        )

    for i in range(num_vertical):
        x = int(width * (i + 1) / (num_vertical + 1))
        svg_lines.append(
            f'<line x1="{x}" y1="20" x2="{x}" y2="{height - 20}" '
            f'stroke="black" stroke-width="{wall_width - 1}"/>'
        )

    svg_lines.append('</svg>')

    return '\n'.join(svg_lines)
