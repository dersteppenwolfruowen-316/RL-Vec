#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rl_vectorizer.data.svgstack_dataset import SVGStackDownloader
from rl_vectorizer.data.tower_dataset import TowerDataset


def download_svgstack(
    output_dir: str = "./data/svgstack",
    splits: List[str] = ["train", "val"],
    max_samples: Optional[Dict[str, int]] = None,
):
    print("=" * 60)
    print("Downloading SVG-Stack Dataset")
    print("=" * 60)

    downloader = SVGStackDownloader(output_dir=output_dir)
    downloader.download_and_prepare(splits=splits, max_samples_per_split=max_samples)

    print("\nValidating dataset...")
    for split in splits:
        stats = downloader.validate_dataset(split)
        print(f"\n{split} split:")
        print(f"  Total: {stats['total']}")
        print(f"  Valid: {stats['valid']}")
        print(f"  Invalid: {stats['invalid']}")
        print(f"  Valid rate: {stats['valid'] / stats['total'] * 100:.2f}%")


def download_tower(
    input_dir: str,
    output_dir: str = "./data/tower",
    extractor: str = "skeleton",
):
    print("=" * 60)
    print("Preparing Tower Dataset")
    print("=" * 60)

    sys.path.insert(0, str(Path(__file__).parent))
    from prepare_tower_data import process_tower_image

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images")

    import json
    samples = []
    for i, image_path in enumerate(tqdm(image_files, desc="Processing")):
        sample = process_tower_image(
            image_path=str(image_path),
            output_dir=output_path,
            extractor=extractor,
            sample_id=i,
        )
        if sample:
            samples.append(sample)

    metadata_path = output_path / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"\nProcessed {len(samples)} samples")
    print(f"Output: {output_path}")


def download_resplan(output_dir: str = "./data/resplan"):
    print("=" * 60)
    print("Downloading ResPlan Dataset")
    print("=" * 60)

    script_path = Path(__file__).parent / "prepare_resplan.sh"
    print(f"Run: bash {script_path}")
    print("\nManual download required from:")
    print("https://github.com/m-agour/ResPlan/releases/download/v1.0/resplan_v1.tar.gz")


def main():
    parser = argparse.ArgumentParser(description="Data downloader for RL Vectorizer")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["svgstack", "tower", "resplan", "all"],
                        help="Dataset to download")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Output directory for data")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Input directory (for tower dataset)")
    parser.add_argument("--max-train", type=int, default=100000,
                        help="Maximum training samples")
    parser.add_argument("--max-val", type=int, default=5000,
                        help="Maximum validation samples")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ["svgstack", "all"]:
        download_svgstack(
            output_dir=str(output_dir / "svgstack"),
            max_samples={"train": args.max_train, "val": args.max_val}
        )

    if args.dataset in ["tower", "all"]:
        if not args.input_dir:
            print("\nSkipping tower dataset (--input-dir required)")
        else:
            download_tower(
                input_dir=args.input_dir,
                output_dir=str(output_dir / "tower"),
            )

    if args.dataset in ["resplan", "all"]:
        download_resplan(output_dir=str(output_dir / "resplan"))

    print("\n" + "=" * 60)
    print("Data download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
