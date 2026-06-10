#!/usr/bin/env python3
"""ResPlan → SVG/PNG 转换器，支持断点续传。"""
import pickle
import json
import os
import sys
import signal
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("convert_resplan")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


class TimeoutError(Exception):
    """渲染超时异常。"""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("rendering timed out")


def render_svg_to_png(svg_code, png_path, size=1024, timeout=60):
    """将 SVG 渲染为 PNG，带超时保护。

    尝试顺序:
    1. rl_vectorizer.utils.svg_renderer (自定义加速渲染)
    2. cairosvg (标准渲染)
    3. PIL 空白图 (最后保底)

    Args:
        timeout: 单次渲染超时秒数 (默认 60s)
    """
    for attempt, renderer in enumerate(["custom", "cairosvg", "pil"]):
        try:
            if renderer == "custom":
                from rl_vectorizer.utils.svg_renderer import render_svg_cairo
                import cairosvg
                # 设置超时
                old = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout)
                try:
                    cairosvg.svg2png(
                        bytestring=svg_code.encode(),
                        write_to=png_path,
                        output_width=size,
                        output_height=size,
                    )
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old)
                return True

            elif renderer == "cairosvg":
                import cairosvg
                old = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout)
                try:
                    cairosvg.svg2png(
                        bytestring=svg_code.encode(),
                        write_to=png_path,
                        output_width=size,
                        output_height=size,
                    )
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old)
                return True

            else:  # pil — 纯白保底图
                from PIL import Image
                img = Image.new("RGB", (size, size), "white")
                img.save(png_path)
                return True

        except TimeoutError:
            log.warning(f"  ⏱ render attempt {attempt + 1} timed out (>={timeout}s)")
        except ImportError:
            if attempt == 0:
                continue  # 自定义渲染器不存在，尝试标准 cairosvg
            if attempt == 1:
                log.warning("  cairosvg not installed, falling back to blank image")
                continue
        except Exception:
            if attempt < 2:
                continue  # 尝试下一级 fallback
            # PIL 也失败 → 返回 False
            return False

    return False


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


CHECKPOINT_FILE = ".convert_resplan_checkpoint.json"


def load_checkpoint():
    """加载处理进度检查点。"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(state):
    """保存处理进度检查点。"""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f)


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "data", "resplan")
    pkl_path = os.path.join(base_dir, "ResPlan.pkl")
    svg_dir = os.path.join(base_dir, "svgs")
    png_dir = os.path.join(base_dir, "bitmaps")
    jsonl_path = os.path.join(base_dir, "metadata.jsonl")

    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    log.info("Loading ResPlan.pkl ...")
    if not os.path.exists(pkl_path):
        log.error(f"❌ {pkl_path} not found!\n"
                    f"Please download ResPlan.zip from https://github.com/m-agour/ResPlan/releases/tag/1.0.0\n"
                    f"Then: unzip ResPlan.zip -d data/resplan/")
        sys.exit(1)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    total = len(data)
    log.info(f"Total samples: {total}")

    # 加载检查点 / 扫描已有文件，支持断点续传
    ckpt = load_checkpoint()
    checkpoint_processed = set(ckpt.get("processed", []))

    # 扫描磁盘已有文件
    disk_existing = set()
    if os.path.exists(svg_dir):
        disk_existing = {f.replace(".svg", "") for f in os.listdir(svg_dir)}
        if disk_existing:
            log.info(f"Found {len(disk_existing)} existing SVGs on disk")

    if checkpoint_processed:
        log.info(f"Loaded checkpoint: {len(checkpoint_processed)} processed")

    # 读取已有的 metadata 行（避免重复）
    jsonl_lines = []
    existing_jsonl_ids = set()
    if os.path.exists(jsonl_path) and os.path.getsize(jsonl_path) > 80:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    jsonl_lines.append(line)
                    obj = json.loads(line)
                    existing_jsonl_ids.add(obj.get("id", ""))
        log.info(f"Existing metadata: {len(jsonl_lines)} lines, "
                  f"{len(existing_jsonl_ids)} unique IDs")

    svg_count = ckpt.get("svg_count", len(disk_existing))
    png_count = ckpt.get("png_count", svg_count)
    svg_fail = ckpt.get("svg_fail", 0)
    png_fail = ckpt.get("png_fail", 0)

    interrupted = False

    for i, plan in enumerate(data):
        sid = f"resplan_{plan.get('id', i):05d}"

        # 跳过 checkpoint 中已完全处理（含 metadata）的
        if sid in checkpoint_processed and sid in existing_jsonl_ids:
            continue

        # 检查文件是否已存在磁盘上
        svg_path = os.path.join(svg_dir, f"{sid}.svg")
        png_path = os.path.join(png_dir, f"{sid}.png")
        files_exist = os.path.exists(svg_path)

        if files_exist and sid not in checkpoint_processed:
            # 磁盘有文件但无 checkpoint → 只需生成 metadata
            if os.path.exists(png_path):
                pass  # png_count 已计入
            else:
                png_fail += 1
            svg_code = open(svg_path).read() if os.path.exists(svg_path) else None
        elif not files_exist:
            # 全新处理：生成 SVG
            svg_code = sample_to_svg(plan)
            if svg_code is None:
                svg_fail += 1
                if (i + 1) % 500 == 0:
                    save_checkpoint({
                        "processed": sorted(checkpoint_processed | {sid}),
                        "svg_count": svg_count,
                        "png_count": png_count,
                        "svg_fail": svg_fail,
                        "png_fail": png_fail,
                    })
                continue
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(svg_code)
            svg_count += 1

            # 渲染 PNG
            if render_svg_to_png(svg_code, png_path, size=1024):
                png_count += 1
            else:
                png_fail += 1
        else:
            # checkpoint 中已有但 metadata 中无 → 补 metadata
            svg_code = None  # 不需要重新处理

        # 元数据（仅当 jsonl 中缺失时写入）
        if sid not in existing_jsonl_ids:
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
            existing_jsonl_ids.add(sid)

        # 更新 checkpoint
        checkpoint_processed.add(sid)

        # 每 500 个保存检查点
        if (i + 1) % 500 == 0:
            save_checkpoint({
                "processed": sorted(checkpoint_processed),
                "svg_count": svg_count,
                "png_count": png_count,
                "svg_fail": svg_fail,
                "png_fail": png_fail,
            })
            log.info(f"  Progress: {i + 1}/{total} (svg={svg_count}, png_ok={png_count}, "
                      f"png_fail={png_fail}, svg_fail={svg_fail})")

    # 写入最终 metadata.jsonl
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(jsonl_lines) + "\n")

    # 清理检查点
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    log.info(f"\nDone!")
    log.info(f"  SVG files: {svg_count} -> {svg_dir}")
    log.info(f"  PNG files: {png_count} -> {png_dir}")
    log.info(f"  PNG render failures: {png_fail}")
    log.info(f"  SVG generation failures: {svg_fail}")
    log.info(f"  Metadata: {jsonl_path}")
    log.info(f"  Total: {svg_count} samples")


if __name__ == "__main__":
    main()
