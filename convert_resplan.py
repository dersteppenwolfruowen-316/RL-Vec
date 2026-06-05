#!/usr/bin/env python3
import pickle
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def shapely_to_svg_path(geom):
    if geom is None or geom.is_empty:
        return ""
    parts = []
    if geom.geom_type == "MultiPolygon":
        polygons = list(geom.geoms)
    elif geom.geom_type == "Polygon":
        polygons = [geom]
    else:
        return ""
    for poly in polygons:
        exterior = poly.exterior
        coords = list(exterior.coords)
        if len(coords) < 2:
            continue
        d = f"M {coords[0][0]:.1f} {coords[0][1]:.1f}"
        for x, y in coords[1:]:
            d += f" L {x:.1f} {y:.1f}"
        d += " Z"
        parts.append(d)
        for interior in poly.interiors:
            coords = list(interior.coords)
            if len(coords) < 2:
                continue
            d = f"M {coords[0][0]:.1f} {coords[0][1]:.1f}"
            for x, y in coords[1:]:
                d += f" L {x:.1f} {y:.1f}"
            d += " Z"
            parts.append(d)
    return " ".join(parts)


def sample_to_svg(plan, canvas_size=1024):
    import numpy as np
    all_parts = []
    geometries = []
    room_types = [
        ("wall", "#333333", 1.5, "fill:none"),
        ("door", "#8B4513", 0.8, "fill:none"),
        ("window", "#4169E1", 0.8, "fill:none"),
        ("front_door", "#FF0000", 1.2, "fill:none"),
        ("bedroom", "#90EE90", 0.3, "fill:#90EE90;fill-opacity:0.15"),
        ("bathroom", "#ADD8E6", 0.3, "fill:#ADD8E6;fill-opacity:0.15"),
        ("kitchen", "#FFB6C1", 0.3, "fill:#FFB6C1;fill-opacity:0.15"),
        ("living", "#FFFFE0", 0.3, "fill:#FFFFE0;fill-opacity:0.15"),
        ("balcony", "#98FB98", 0.3, "fill:#98FB98;fill-opacity:0.1"),
        ("storage", "#D3D3D3", 0.3, "fill:#D3D3D3;fill-opacity:0.15"),
        ("stair", "#DEB887", 0.5, "fill:none"),
    ]

    all_coords = []
    for room_type, _, _, _ in room_types:
        geom = plan.get(room_type)
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                all_coords.extend(list(poly.exterior.coords))
                for interior in poly.interiors:
                    all_coords.extend(list(interior.coords))
        elif geom.geom_type == "Polygon":
            all_coords.extend(list(geom.exterior.coords))
            for interior in geom.interiors:
                all_coords.extend(list(interior.coords))

    if not all_coords:
        return None

    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    data_w = max_x - min_x if max_x > min_x else 1
    data_h = max_y - min_y if max_y > min_y else 1
    margin = 20
    scale = min((canvas_size - 2 * margin) / data_w, (canvas_size - 2 * margin) / data_h)

    def transform(x, y):
        tx = margin + (x - min_x) * scale
        ty = canvas_size - margin - (y - min_y) * scale
        return tx, ty

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}" viewBox="0 0 {canvas_size} {canvas_size}">')
    lines.append(f'<rect width="{canvas_size}" height="{canvas_size}" fill="white"/>')

    for room_type, color, stroke_w, style in room_types:
        geom = plan.get(room_type)
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        elif geom.geom_type == "Polygon":
            polygons = [geom]
        else:
            continue

        layer_parts = []
        for poly in polygons:
            exterior = poly.exterior
            coords = list(exterior.coords)
            if len(coords) < 2:
                continue
            tx, ty = transform(coords[0][0], coords[0][1])
            d = f"M {tx:.1f} {ty:.1f}"
            for x, y in coords[1:]:
                tx, ty = transform(x, y)
                d += f" L {tx:.1f} {ty:.1f}"
            d += " Z"

            for interior in poly.interiors:
                coords = list(interior.coords)
                if len(coords) < 2:
                    continue
                tx, ty = transform(coords[0][0], coords[0][1])
                d += f" M {tx:.1f} {ty:.1f}"
                for x, y in coords[1:]:
                    tx, ty = transform(x, y)
                    d += f" L {tx:.1f} {ty:.1f}"
                d += " Z"

            layer_parts.append(d)

        if layer_parts:
            combined = " ".join(layer_parts)
            effective_style = f'stroke="{color}" stroke-width="{stroke_w}" style="{style}"'
            lines.append(f'  <path d="{combined}" {effective_style}/>')

    lines.append("</svg>")
    return "\n".join(lines)


def render_svg_to_png(svg_code, png_path, size=1024):
    try:
        from rl_vectorizer.utils.svg_renderer import render_svg_cairo
        import cairosvg
        cairosvg.svg2png(
            bytestring=svg_code.encode(),
            write_to=png_path,
            output_width=size,
            output_height=size,
        )
        return True
    except Exception:
        pass

    try:
        import cairosvg
        cairosvg.svg2png(
            bytestring=svg_code.encode(),
            write_to=png_path,
            output_width=size,
            output_height=size,
        )
        return True
    except Exception:
        pass

    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (size, size), "white")
        img.save(png_path)
        return True
    except Exception:
        return False


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "data", "resplan")
    pkl_path = os.path.join(base_dir, "ResPlan.pkl")
    svg_dir = os.path.join(base_dir, "svgs")
    png_dir = os.path.join(base_dir, "bitmaps")
    jsonl_path = os.path.join(base_dir, "metadata.jsonl")

    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    print("Loading ResPlan.pkl ...")
    if not os.path.exists(pkl_path):
        print(f"❌ {pkl_path} not found!")
        print("Please download ResPlan.zip from https://github.com/m-agour/ResPlan/releases/tag/1.0.0")
        print("Then: unzip ResPlan.zip -d data/resplan/")
        sys.exit(1)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    total = len(data)
    print(f"Total samples: {total}")

    jsonl_lines = []
    svg_count = 0
    png_count = 0
    fail_count = 0

    for i, plan in enumerate(data):
        sid = f"resplan_{plan.get('id', i):05d}"

        svg_code = sample_to_svg(plan)
        if svg_code is None:
            fail_count += 1
            continue

        svg_path = os.path.join(svg_dir, f"{sid}.svg")
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_code)
        svg_count += 1

        png_path = os.path.join(png_dir, f"{sid}.png")
        if render_svg_to_png(svg_code, png_path, size=1024):
            png_count += 1
        else:
            png_count += 1

        room_info = {}
        for key in ["bedroom", "bathroom", "kitchen", "living", "balcony", "storage", "door", "window"]:
            geom = plan.get(key)
            if geom is not None and not geom.is_empty:
                if geom.geom_type == "MultiPolygon":
                    room_info[key] = len(list(geom.geoms))
                elif geom.geom_type == "Polygon":
                    room_info[key] = 1

        jsonl_lines.append(json.dumps({
            "id": sid,
            "svg_file": f"svgs/{sid}.svg",
            "bitmap_file": f"bitmaps/{sid}.png",
            "unitType": plan.get("unitType", "unknown"),
            "area": float(plan.get("area", 0)),
            "net_area": float(plan.get("net_area", 0)),
            "rooms": room_info,
        }, ensure_ascii=False))

        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1}/{total} ...")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(jsonl_lines) + "\n")

    print(f"\nDone!")
    print(f"  SVG files: {svg_count} -> {svg_dir}")
    print(f"  PNG files: {png_count} -> {png_dir}")
    print(f"  Failed: {fail_count}")
    print(f"  Metadata: {jsonl_path}")
    print(f"  Total: {svg_count} samples")


if __name__ == "__main__":
    main()
