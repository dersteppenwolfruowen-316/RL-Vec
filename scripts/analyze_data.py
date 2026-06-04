#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm


class DatasetAnalyzer:
    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.metadata_path = self.data_dir / f"{split}_metadata.jsonl"
        self.samples: List[Dict] = []

    def load(self):
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {self.metadata_path}")

        with open(self.metadata_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

        print(f"Loaded {len(self.samples)} samples from {self.split}")

    def analyze_svg_complexity(self) -> Dict[str, Any]:
        from lxml import etree

        line_counts = []
        path_counts = []
        total_elements = []
        svg_lengths = []

        for sample in tqdm(self.samples, desc="Analyzing SVG complexity"):
            svg_code = sample.get("svg", "")
            if not svg_code:
                continue

            svg_lengths.append(len(svg_code))

            try:
                tree = etree.fromstring(svg_code.encode())
                ns = {"svg": "http://www.w3.org/2000/svg"}

                lines = tree.xpath("//svg:line", namespaces=ns)
                paths = tree.xpath("//svg:path", namespaces=ns)
                rects = tree.xpath("//svg:rect", namespaces=ns)

                line_counts.append(len(lines))
                path_counts.append(len(paths))
                total_elements.append(len(list(tree.iter())))
            except Exception:
                pass

        return {
            "line_count": {
                "mean": np.mean(line_counts) if line_counts else 0,
                "std": np.std(line_counts) if line_counts else 0,
                "min": np.min(line_counts) if line_counts else 0,
                "max": np.max(line_counts) if line_counts else 0,
                "median": np.median(line_counts) if line_counts else 0,
            },
            "path_count": {
                "mean": np.mean(path_counts) if path_counts else 0,
                "std": np.std(path_counts) if path_counts else 0,
                "min": np.min(path_counts) if path_counts else 0,
                "max": np.max(path_counts) if path_counts else 0,
            },
            "total_elements": {
                "mean": np.mean(total_elements) if total_elements else 0,
                "std": np.std(total_elements) if total_elements else 0,
            },
            "svg_length": {
                "mean": np.mean(svg_lengths) if svg_lengths else 0,
                "std": np.std(svg_lengths) if svg_lengths else 0,
            },
        }

    def analyze_caption_statistics(self) -> Dict[str, Any]:
        caption_lengths = []
        empty_captions = 0

        for sample in self.samples:
            caption = sample.get("caption", "") or sample.get("description", "")
            if not caption:
                empty_captions += 1
            caption_lengths.append(len(caption.split()))

        return {
            "total": len(self.samples),
            "empty_captions": empty_captions,
            "caption_length": {
                "mean": np.mean(caption_lengths) if caption_lengths else 0,
                "std": np.std(caption_lengths) if caption_lengths else 0,
                "min": np.min(caption_lengths) if caption_lengths else 0,
                "max": np.max(caption_lengths) if caption_lengths else 0,
                "median": np.median(caption_lengths) if caption_lengths else 0,
            },
        }

    def validate_svg_syntax(self) -> Dict[str, Any]:
        from lxml import etree

        valid_count = 0
        invalid_count = 0
        error_types = Counter()

        for sample in tqdm(self.samples, desc="Validating SVG syntax"):
            svg_code = sample.get("svg", "")
            if not svg_code:
                invalid_count += 1
                error_types["empty"] += 1
                continue

            try:
                etree.fromstring(svg_code.encode())
                valid_count += 1
            except etree.XMLSyntaxError as e:
                invalid_count += 1
                error_types[str(type(e).__name__)] += 1
            except Exception as e:
                invalid_count += 1
                error_types["other"] += 1

        return {
            "valid": valid_count,
            "invalid": invalid_count,
            "validity_rate": valid_count / (valid_count + invalid_count) if (valid_count + invalid_count) > 0 else 0,
            "error_types": dict(error_types),
        }

    def analyze_data_distribution(self) -> Dict[str, Any]:
        line_count_distribution = defaultdict(int)
        for sample in self.samples:
            svg = sample.get("svg", "")
            line_count = svg.count("<line") + svg.count("<path")
            if line_count < 10:
                bucket = "0-10"
            elif line_count < 50:
                bucket = "10-50"
            elif line_count < 100:
                bucket = "50-100"
            elif line_count < 500:
                bucket = "100-500"
            else:
                bucket = "500+"
            line_count_distribution[bucket] += 1

        return {
            "line_count_distribution": dict(line_count_distribution),
        }

    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        report = {
            "split": self.split,
            "total_samples": len(self.samples),
            "svg_complexity": self.analyze_svg_complexity(),
            "caption_statistics": self.analyze_caption_statistics(),
            "svg_validation": self.validate_svg_syntax(),
            "data_distribution": self.analyze_data_distribution(),
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {output_path}")

        return report


def print_report(report: Dict[str, Any]):
    print("\n" + "=" * 60)
    print("Dataset Analysis Report")
    print("=" * 60)

    print(f"\nTotal samples: {report['total_samples']}")

    print("\n--- SVG Validation ---")
    validation = report["svg_validation"]
    print(f"Valid SVG: {validation['valid']}")
    print(f"Invalid SVG: {validation['invalid']}")
    print(f"Validity rate: {validation['validity_rate']:.2%}")
    if validation["error_types"]:
        print(f"Error types: {validation['error_types']}")

    print("\n--- SVG Complexity ---")
    complexity = report["svg_complexity"]
    print(f"Line count: mean={complexity['line_count']['mean']:.1f}, "
          f"std={complexity['line_count']['std']:.1f}, "
          f"median={complexity['line_count']['median']:.1f}")
    print(f"Path count: mean={complexity['path_count']['mean']:.1f}, "
          f"std={complexity['path_count']['std']:.1f}")

    print("\n--- Caption Statistics ---")
    caption = report["caption_statistics"]
    print(f"Empty captions: {caption['empty_captions']}")
    print(f"Caption length: mean={caption['caption_length']['mean']:.1f} words, "
          f"median={caption['caption_length']['median']:.1f}")

    print("\n--- Line Count Distribution ---")
    dist = report["data_distribution"]["line_count_distribution"]
    for bucket, count in sorted(dist.items()):
        percentage = count / report["total_samples"] * 100
        print(f"  {bucket}: {count} ({percentage:.1f}%)")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset statistics")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--output", type=str, default=None, help="Output report path")
    args = parser.parse_args()

    analyzer = DatasetAnalyzer(args.data_dir, args.split)
    analyzer.load()
    report = analyzer.generate_report(args.output)
    print_report(report)


if __name__ == "__main__":
    main()
