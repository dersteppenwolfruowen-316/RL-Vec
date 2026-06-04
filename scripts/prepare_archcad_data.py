import os
import argparse
from pathlib import Path

from datasets import Dataset


def collect_samples(data_dir: Path) -> list[dict]:
    png_dir = data_dir / "png"
    svg_dir = data_dir / "svg"

    if not png_dir.exists() or not svg_dir.exists():
        raise FileNotFoundError(
            f"Expected {png_dir} and {svg_dir} to exist. "
            "Run `python download_archcad.py` first."
        )

    png_files = sorted(png_dir.rglob("*.png"))
    print(f"  Found {len(png_files)} PNG files")

    samples = []
    skipped = 0
    for png_path in png_files:
        rel = png_path.relative_to(png_dir)
        svg_path = svg_dir / rel.with_suffix(".svg")

        if not svg_path.exists():
            skipped += 1
            continue

        svg_code = svg_path.read_text(encoding="utf-8")
        if not svg_code.strip():
            skipped += 1
            continue

        sample_id = rel.stem
        prompt_text = (
            "Convert this architectural CAD drawing into SVG code. "
            "Generate precise SVG with correct line positions, "
            "semantic labels, and geometric primitives "
            "(lines, arcs, circles) using appropriate attributes."
        )

        samples.append({
            "prompt": f"<image>\n{prompt_text}",
            "images": str(png_path),
            "ground_truth": svg_code,
            "data_source": "archcad",
            "id": sample_id,
        })

    print(f"  Valid samples: {len(samples)}, skipped: {skipped}")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Convert ArchCAD data to EasyR1 HuggingFace format"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/archcad/data",
        help="ArchCAD data directory containing png/ and svg/",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/easyr1_archcad",
        help="Output directory for HuggingFace dataset",
    )
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push to HuggingFace Hub")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting ArchCAD samples ...")
    samples = collect_samples(data_dir)

    split_idx = int(len(samples) * args.train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    if args.max_train and len(train_samples) > args.max_train:
        train_samples = train_samples[:args.max_train]
    if args.max_val and len(val_samples) > args.max_val:
        val_samples = val_samples[:args.max_val]

    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = Dataset.from_list(train_samples)
    val_ds = Dataset.from_list(val_samples)

    train_ds.save_to_disk(str(output_dir / "train"))
    val_ds.save_to_disk(str(output_dir / "val"))

    if args.push_to_hub:
        train_ds.push_to_hub("user/archcad", split="train", private=True)
        val_ds.push_to_hub("user/archcad", split="val", private=True)

    print(f"\nDone! Saved to {output_dir}/")
    print(f"  Train: {output_dir}/train")
    print(f"  Val:   {output_dir}/val")
    print(f"\nTo use with EasyR1:")
    print(f"  data.train_files={output_dir}/train")
    print(f"  data.val_files={output_dir}/val")


if __name__ == "__main__":
    main()
