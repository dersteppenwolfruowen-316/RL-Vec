import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rl_vectorizer.utils.svg_validator import validate_svg, extract_svg_stats
from rl_vectorizer.utils.svg_renderer import render_svg_cairo


def optimize_for_cad(svg_code: str) -> str:
    import re

    svg_code = re.sub(r'\s+', ' ', svg_code)
    svg_code = re.sub(r'\s*>\s*<', '><', svg_code)
    svg_code = re.sub(r'id="[^"]*"', '', svg_code)
    svg_code = re.sub(r'class="[^"]*"', '', svg_code)

    svg_code = svg_code.replace('stroke-width="1.5"', 'stroke-width="0.5"')
    svg_code = svg_code.replace('stroke-width="2"', 'stroke-width="0.5"')
    svg_code = svg_code.replace('stroke-width="3"', 'stroke-width="0.5"')

    return svg_code


def convert_svg_to_dxf(svg_path: str, output_path: str) -> bool:
    try:
        import svg2dxf
        svg2dxf.convert(svg_path, output_path)
        return True
    except ImportError:
        print("svg2dxf not installed. Install with: pip install svg2dxf")
        return False
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Post-process SVG for CAD import")
    parser.add_argument("--input", type=str, required=True, help="Input SVG file or directory")
    parser.add_argument("--output", type=str, default="./cad_output", help="Output directory")
    parser.add_argument("--format", type=str, default="svg", choices=["svg", "dxf"], help="Output format")
    parser.add_argument("--optimize", action="store_true", help="Optimize for CAD")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    input_path = Path(args.input)

    svg_files = []
    if input_path.is_file():
        svg_files = [input_path]
    elif input_path.is_dir():
        svg_files = list(input_path.glob("*.svg"))

    print(f"Found {len(svg_files)} SVG files")

    success_count = 0
    for svg_path in svg_files:
        with open(svg_path, "r") as f:
            svg_code = f.read()

        if not validate_svg(svg_code):
            print(f"Invalid SVG: {svg_path}")
            continue

        if args.optimize:
            svg_code = optimize_for_cad(svg_code)

        output_path = Path(args.output) / svg_path.name
        with open(output_path, "w") as f:
            f.write(svg_code)

        if args.format == "dxf":
            dxf_path = output_path.with_suffix(".dxf")
            if convert_svg_to_dxf(str(output_path), str(dxf_path)):
                print(f"Converted: {svg_path.name} -> {dxf_path.name}")

        success_count += 1
        print(f"Processed: {svg_path.name}")

    print(f"\nSuccess: {success_count}/{len(svg_files)} files")


if __name__ == "__main__":
    main()
