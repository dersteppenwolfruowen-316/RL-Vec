#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SVG-Stack dataset")
    parser.add_argument("--output-dir", type=str, default="./data/svgstack", help="Output directory")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--max-train", type=int, default=100000, help="Max training samples")
    parser.add_argument("--max-val", type=int, default=5000, help="Max validation samples")
    parser.add_argument("--max-test", type=int, default=2000, help="Max test samples")
    parser.add_argument("--render-size", type=int, default=512, help="SVG render size")
    parser.add_argument("--validate", action="store_true", help="Validate SVG syntax")
    parser.add_argument("--deduplicate", action="store_true", help="Remove duplicate SVGs")
    return parser.parse_args()


def download_from_huggingface(output_dir: str, max_samples: Optional[Dict[str, int]] = None):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading starvector/svg-stack from HuggingFace...")
    ds = load_dataset("starvector/svg-stack", trust_remote_code=True)

    splits = ["train", "validation", "test"]
    if max_samples is None:
        max_samples = {}

    for split in splits:
        if split not in ds:
            continue

        split_ds = ds[split]
        max_n = max_samples.get(split, len(split_ds))
        max_n = min(max_n, len(split_ds))

        print(f"\nProcessing split '{split}' with {max_n} samples...")

        metadata_path = output_path / f"{split}_metadata.jsonl"
        with open(metadata_path, "w") as f:
            for i, sample in enumerate(tqdm(split_ds, desc=f"Processing {split}")):
                if i >= max_n:
                    break

                sample_id = sample.get("Filename", f"{split}_{i}")

                svg_code = sample["Svg"]
                caption = sample.get("Captions", "")

                metadata = {
                    "id": sample_id,
                    "svg": svg_code,
                    "caption": caption,
                }

                f.write(json.dumps(metadata) + "\n")

        print(f"Saved {i + 1} samples to {metadata_path}")

    return output_path


def validate_svg(svg_code: str) -> bool:
    from lxml import etree
    try:
        etree.fromstring(svg_code.encode())
        return True
    except Exception:
        return False


def deduplicate_svg(svg_codes: List[str]) -> List[str]:
    seen = set()
    unique = []

    for svg in svg_codes:
        normalized = svg.lower().replace(" ", "").replace("\n", "")
        if normalized not in seen:
            seen.add(normalized)
            unique.append(svg)

    return unique


def filter_invalid_svg(input_path: Path, output_path: Path, split: str):
    metadata_path = input_path / f"{split}_metadata.jsonl"
    if not metadata_path.exists():
        print(f"Metadata not found: {metadata_path}")
        return

    valid_path = output_path / f"{split}_valid.jsonl"
    invalid_path = output_path / f"{split}_invalid.jsonl"

    valid_count = 0
    invalid_count = 0

    with open(metadata_path, "r") as f_in, \
         open(valid_path, "w") as f_valid, \
         open(invalid_path, "w") as f_invalid:

        for line in tqdm(f_in, desc=f"Validating {split}"):
            data = json.loads(line.strip())
            svg_code = data.get("svg", "")

            if validate_svg(svg_code):
                f_valid.write(json.dumps(data) + "\n")
                valid_count += 1
            else:
                f_invalid.write(json.dumps(data) + "\n")
                invalid_count += 1

    print(f"\n{split} split:")
    print(f"  Valid: {valid_count}")
    print(f"  Invalid: {invalid_count}")
    print(f"  Valid rate: {valid_count / (valid_count + invalid_count) * 100:.2f}%")

    valid_path.replace(output_path / f"{split}_metadata.jsonl")


def analyze_svgstack(output_dir: str) -> Dict[str, any]:
    output_path = Path(output_dir)
    stats = {}

    for split in ["train", "validation", "test"]:
        metadata_path = output_path / f"{split}_metadata.jsonl"
        if not metadata_path.exists():
            continue

        line_counts = []
        caption_lengths = []
        valid_count = 0
        invalid_count = 0

        with open(metadata_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                svg_code = data.get("svg", "")
                caption = data.get("caption", "")

                if validate_svg(svg_code):
                    valid_count += 1
                    line_counts.append(svg_code.count("<line") + svg_code.count("<path"))
                else:
                    invalid_count += 1

                caption_lengths.append(len(caption.split()))

        import numpy as np

        stats[split] = {
            "total": valid_count + invalid_count,
            "valid": valid_count,
            "invalid": invalid_count,
            "validity_rate": valid_count / (valid_count + invalid_count) if (valid_count + invalid_count) > 0 else 0,
            "line_count_mean": np.mean(line_counts) if line_counts else 0,
            "line_count_std": np.std(line_counts) if line_counts else 0,
            "caption_length_mean": np.mean(caption_lengths) if caption_lengths else 0,
        }

    return stats


def sample_svgstack(output_dir: str, num_samples: int = 100) -> List[Dict]:
    output_path = Path(output_dir)
    metadata_path = output_path / "train_metadata.jsonl"

    if not metadata_path.exists():
        metadata_path = output_path / "train_valid.jsonl"

    samples = []
    with open(metadata_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line.strip()))

    return samples


def main():
    args = parse_args()

    print("=" * 60)
    print("SVG-Stack Dataset Preparation")
    print("=" * 60)

    print("\nStep 1: Downloading from HuggingFace...")
    download_from_huggingface(
        output_dir=args.output_dir,
        max_samples={
            "train": args.max_train,
            "validation": args.max_val,
            "test": args.max_test,
        }
    )

    if args.validate:
        print("\nStep 2: Validating SVG syntax...")
        from pathlib import Path
        filter_invalid_svg(Path(args.output_dir), Path(args.output_dir), "train")
        filter_invalid_svg(Path(args.output_dir), Path(args.output_dir), "validation")
        filter_invalid_svg(Path(args.output_dir), Path(args.output_dir), "test")

    if args.deduplicate:
        print("\nStep 3: Deduplicating SVGs...")
        print("Deduplication not yet implemented")

    print("\nStep 4: Analyzing dataset...")
    stats = analyze_svgstack(args.output_dir)

    for split, stat in stats.items():
        print(f"\n{split}:")
        print(f"  Total: {stat['total']}")
        print(f"  Valid: {stat['valid']} ({stat['validity_rate'] * 100:.2f}%)")
        print(f"  Line count: {stat['line_count_mean']:.1f} ± {stat['line_count_std']:.1f}")
        print(f"  Caption length: {stat['caption_length_mean']:.1f} words")

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print(f"Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
