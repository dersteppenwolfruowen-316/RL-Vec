import os
import json
import argparse
from pathlib import Path

from datasets import Dataset


def load_metadata(metadata_path: str) -> list[dict]:
    samples = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def build_sample(entry: dict, base_dir: Path, image_dir: str | None = None) -> dict:
    svg_path = base_dir / entry["svg_file"]
    png_path = base_dir / entry["bitmap_file"]

    if not svg_path.exists() or not png_path.exists():
        return None

    svg_code = svg_path.read_text(encoding="utf-8")

    room_desc = ", ".join(
        f"{k}={v}" for k, v in entry.get("rooms", {}).items()
    )
    unit_type = entry.get("unitType", "unknown")
    area = entry.get("area", 0)
    prompt_text = (
        f"Convert this architectural floor plan image into SVG code. "
        f"The floor plan is a {unit_type} apartment with area {area:.1f} sqm. "
        f"Room types: {room_desc}. "
        f"Generate precise SVG with correct wall positions, door/window placements, "
        f"and room areas using appropriate fill colors."
    )

    if image_dir:
        image_path = str(Path(image_dir) / entry["bitmap_file"])
    else:
        image_path = str(png_path)

    return {
        "prompt": f"<image>\n{prompt_text}",
        "images": image_path,
        "ground_truth": svg_code,
        "data_source": "resplan",
        "id": entry["id"],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare ResPlan data for EasyR1 training")
    parser.add_argument(
        "--metadata", type=str,
        default="data/resplan/metadata.jsonl",
        help="Path to ResPlan metadata.jsonl",
    )
    parser.add_argument(
        "--base-dir", type=str,
        default="data/resplan",
        help="Base directory containing svgs/ and bitmaps/",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="data/easyr1_resplan",
        help="Output directory for HuggingFace dataset",
    )
    parser.add_argument(
        "--train-ratio", type=float,
        default=0.9,
        help="Train/val split ratio",
    )
    parser.add_argument(
        "--max-train", type=int,
        default=None,
        help="Max training samples",
    )
    parser.add_argument(
        "--max-val", type=int,
        default=None,
        help="Max validation samples",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata from {args.metadata} ...")
    all_entries = load_metadata(args.metadata)
    print(f"  Total entries: {len(all_entries)}")

    print("Building samples ...")
    samples = []
    for entry in all_entries:
        sample = build_sample(entry, base_dir)
        if sample is not None:
            samples.append(sample)
    print(f"  Valid samples: {len(samples)}")

    split_idx = int(len(samples) * args.train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    if args.max_train and len(train_samples) > args.max_train:
        train_samples = train_samples[: args.max_train]
    if args.max_val and len(val_samples) > args.max_val:
        val_samples = val_samples[: args.max_val]

    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

    print("Creating HuggingFace datasets ...")
    train_ds = Dataset.from_list(train_samples)
    val_ds = Dataset.from_list(val_samples)

    train_ds.save_to_disk(str(output_dir / "train"))
    val_ds.save_to_disk(str(output_dir / "val"))

    train_ds.push_to_hub(
        f"user/resplan",
        split="train",
        private=True,
    )
    val_ds.push_to_hub(
        f"user/resplan",
        split="val",
        private=True,
    )

    print(f"\nDone! Dataset saved to {output_dir}/")
    print(f"  Train: {output_dir}/train")
    print(f"  Val:   {output_dir}/val")
    print(f"\nTo use with EasyR1:")
    print(f"  data.train_files={output_dir}/train")
    print(f"  data.val_files={output_dir}/val")


if __name__ == "__main__":
    main()
