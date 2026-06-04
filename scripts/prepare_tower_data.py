#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare tower drawing data")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing tower images")
    parser.add_argument("--output", type=str, default="./data/tower", help="Output directory")
    parser.add_argument("--extractor", type=str, default="skeleton", choices=["skeleton", "baseline"], help="Extraction method")
    parser.add_argument("--vlm-api", type=str, default=None, help="VLM API for description (optional)")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to process")
    return parser.parse_args()


def extract_skeleton_svg(image_path: str) -> str:
    try:
        import cv2
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""

        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        skeleton = cv2.ximgproc.thinning(binary)

        lines = []
        height, width = skeleton.shape
        for y in range(0, height, 10):
            for x in range(0, width, 10):
                if skeleton[y, x] > 0:
                    line_length = 10
                    lines.append(
                        f'<line x1="{x}" y1="{y}" x2="{x + line_length}" y2="{y}" '
                        f'stroke="black" stroke-width="1"/>'
                    )

        svg_code = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        svg_code += "".join(lines)
        svg_code += "</svg>"
        return svg_code
    except ImportError:
        print("opencv-python not installed, using baseline extraction")
        return create_baseline_svg(image_path)


def create_baseline_svg(image_path: str) -> str:
    img = Image.open(image_path)
    width, height = img.size

    lines = [
        f'<line x1="0" y1="{i * height // 10}" x2="{width}" y2="{i * height // 10}" '
        f'stroke="black" stroke-width="1"/>'
        for i in range(1, 10)
    ]
    lines.extend([
        f'<line x1="{i * width // 10}" y1="0" x2="{i * width // 10}" y2="{height}" '
        f'stroke="black" stroke-width="1"/>'
        for i in range(1, 10)
    ])

    svg_code = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
    svg_code += "".join(lines)
    svg_code += "</svg>"
    return svg_code


def get_image_description(image_path: str, vlm_api: Optional[str] = None) -> str:
    if vlm_api is None:
        return "Tower drawing with structural members"

    try:
        import requests
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"prompt": "Describe the main structural features of this tower drawing"}
            response = requests.post(vlm_api, files=files, data=data, timeout=30)
            if response.status_code == 200:
                return response.json().get("description", "")
    except Exception as e:
        print(f"VLM API error: {e}")

    return "Tower drawing with structural members"


def process_tower_image(
    image_path: str,
    output_dir: Path,
    extractor: str = "skeleton",
    vlm_api: Optional[str] = None,
    sample_id: int = 0,
) -> Optional[Dict]:
    try:
        img = Image.open(image_path)
        width, height = img.size

        if extractor == "skeleton":
            svg_code = extract_skeleton_svg(image_path)
        else:
            svg_code = create_baseline_svg(image_path)

        if not svg_code:
            return None

        description = get_image_description(image_path, vlm_api)

        svg_output_path = output_dir / "svg" / f"tower_{sample_id:04d}.svg"
        svg_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(svg_output_path, "w") as f:
            f.write(svg_code)

        sample = {
            "id": f"tower_{sample_id:04d}",
            "image": str(image_path),
            "image_width": width,
            "image_height": height,
            "svg": svg_code,
            "description": description,
            "metadata": {
                "line_count": svg_code.count("<line"),
                "path_count": svg_code.count("<path"),
                "source": "tower_drawing",
                "extractor": extractor,
            },
        }

        return sample

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No images found in {input_path}")
        sys.exit(1)

    print(f"Found {len(image_files)} images")

    if args.max_images:
        image_files = image_files[: args.max_images]

    samples = []
    for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        sample = process_tower_image(
            image_path=str(image_path),
            output_dir=output_dir,
            extractor=args.extractor,
            vlm_api=args.vlm_api,
            sample_id=i,
        )
        if sample:
            samples.append(sample)

    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    stats = {
        "total_images": len(image_files),
        "processed_samples": len(samples),
        "output_dir": str(output_dir),
    }
    print(f"\nProcessing complete:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Processed: {stats['processed_samples']}")
    print(f"  Output: {stats['output_dir']}")


if __name__ == "__main__":
    main()
