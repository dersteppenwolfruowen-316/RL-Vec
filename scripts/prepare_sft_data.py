"""ResPlan → SFT 中间指令训练数据转换器

将 ResPlan 的 SVG + 元数据转换为结构化的中间指令格式（方案 A），
用于 VLM 的 SFT 阶段训练。

输出格式 (JSONL):
  {
    "id": "resplan_00013",
    "image": "bitmaps/resplan_00013.png",
    "conversations": [
      {"from": "user",  "value": "<image>\nConvert this architectural floor plan to SVG format."},
      {"from": "assistant", "value": "<analysis>...</analysis>\n<outer_wall>...</outer_wall>\n...\n<svg>...</svg>"}
    ],
    "metadata": { ... }
  }
"""
import sys, os, json, re
from pathlib import Path
from lxml import etree
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union

# 颜色 → 语义映射
STROKE_LABELS = {
    "#333333": "wall",
    "#8B4513": "door",
    "#4169E1": "window",
    "#FF0000": "front_door",
}

FILL_LABELS = {
    "#90EE90": "bedroom",
    "#ADD8E6": "bathroom",
    "#FFB6C1": "kitchen",
    "#FFFFE0": "living_room",
    "#98FB98": "balcony",
    "#D3D3D3": "storage",
    "#DEB887": "stair",
}

SVG_NSMAP = {"svg": "http://www.w3.org/2000/svg"}
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "resplan"


def parse_path_d(d: str) -> list:
    """将 SVG path 的 d 属性解析为独立的子路径列表。
    每个子路径用 (cmd, coords_list) 表示。
    """
    tokens = re.findall(r"[MZLHVCSQTAmlhvcsqtaz]|-?\d+\.?\d*", d)
    paths = []
    i = 0
    while i < len(tokens):
        cmd = tokens[i]
        if cmd in "Mm":
            i += 1
            nums = []
            while i < len(tokens) and re.match(r"^-?\d+\.?\d*$", tokens[i]):
                nums.append(float(tokens[i]))
                i += 1
            paths.append({"type": "move", "cmd": cmd, "coords": nums})
        elif cmd in "Ll":
            i += 1
            nums = []
            while i < len(tokens) and re.match(r"^-?\d+\.?\d*$", tokens[i]):
                nums.append(float(tokens[i]))
                i += 1
            paths.append({"type": "line", "cmd": cmd, "coords": nums})
        elif cmd in "Zz":
            paths.append({"type": "close", "cmd": cmd})
            i += 1
        else:
            i += 1
    return paths


def path_to_polygons(d: str) -> list:
    """将 SVG path d 属性转换为 Shapely Polygon 列表。
    处理 M... Z 子路径，支持相对/绝对坐标。
    """
    tokens = re.findall(r"[MZLHVCSQTAmlhvcsqtaz]|-?\d+\.?\d*", d)
    polygons = []
    current_ring = []
    cx, cy = 0.0, 0.0
    first_x, first_y = 0.0, 0.0
    i = 0

    def next_coord():
        nonlocal i
        if i < len(tokens) and re.match(r"^-?\d+\.?\d*$", tokens[i]):
            v = float(tokens[i])
            i += 1
            return v
        return 0.0

    while i < len(tokens):
        cmd = tokens[i]
        i += 1

        if cmd == "M":
            if current_ring and len(current_ring) > 2:
                try:
                    polygons.append(Polygon(current_ring))
                except Exception:
                    pass
            current_ring = []
            x = next_coord()
            y = next_coord()
            cx, cy = x, y
            first_x, first_y = x, y
            current_ring.append((x, y))
        elif cmd == "m":
            if current_ring and len(current_ring) > 2:
                try:
                    polygons.append(Polygon(current_ring))
                except Exception:
                    pass
            current_ring = []
            dx = next_coord()
            dy = next_coord()
            cx += dx
            cy += dy
            first_x, first_y = cx, cy
            current_ring.append((cx, cy))
        elif cmd in "Ll":
            x = next_coord()
            y = next_coord()
            if cmd == "l":
                cx += x
                cy += y
            else:
                cx, cy = x, y
            current_ring.append((cx, cy))
        elif cmd in "Hh":
            x = next_coord()
            if cmd == "h":
                cx += x
            else:
                cx = x
            current_ring.append((cx, cy))
        elif cmd in "Vv":
            y = next_coord()
            if cmd == "v":
                cy += y
            else:
                cy = y
            current_ring.append((cx, cy))
        elif cmd in "Zz":
            if current_ring and current_ring[0] != current_ring[-1]:
                current_ring.append((first_x, first_y))
            if current_ring and len(current_ring) > 2:
                try:
                    polygons.append(Polygon(current_ring))
                except Exception:
                    pass
            current_ring = []
        elif cmd in "Cc":
            # 跳过曲线控制点，取终点
            for _ in range(3):
                if cmd == "C":
                    cx = next_coord()
                    cy = next_coord()
                else:
                    cx += next_coord()
                    cy += next_coord()
            current_ring.append((cx, cy))
        elif cmd in "Qq":
            for _ in range(2):
                if cmd == "Q":
                    cx = next_coord()
                    cy = next_coord()
                else:
                    cx += next_coord()
                    cy += next_coord()
            current_ring.append((cx, cy))
        else:
            break

    if current_ring and len(current_ring) > 2:
        try:
            polygons.append(Polygon(current_ring))
        except Exception:
            pass

    return [p for p in polygons if p and not p.is_empty]


def get_fill_color(style: str) -> str:
    """从 style 属性提取 fill 颜色"""
    m = re.search(r"fill:#([0-9A-Fa-f]+)", style)
    if m:
        return "#" + m.group(1).upper()
    return ""


def simplify_path(d: str, max_coords: int = 20) -> str:
    """简化 path 的坐标精度"""
    def round_coord(m):
        val = float(m.group())
        return f"{val:.0f}"
    return re.sub(r"-?\d+\.?\d*", round_coord, d)


def classify_walls(polygons: list) -> tuple:
    """将墙体多边形分类为 outer_wall 和 partition_walls。

    外墙上所有其他墙的边界——即面积最大、包含其他墙的。
    """
    if not polygons:
        return None, []

    # 按面积排序
    sorted_polys = sorted(polygons, key=lambda p: p.area, reverse=True)

    if len(sorted_polys) == 1:
        return sorted_polys[0], []

    outer = sorted_polys[0]
    inners = sorted_polys[1:]

    # 验证 outer 是否真的包含其他墙
    contained = []
    free = []
    for p in inners:
        if outer.contains(p) or outer.covers(p):
            contained.append(p)
        else:
            free.append(p)

    if free:
        # 如果有不被 outer 包含的墙，外扩 outer
        all_polys = sorted_polys
        all_boxes = [p.envelope for p in all_polys]
        outer = unary_union(all_boxes).envelope
        contained = all_polys[1:]

    return outer, contained


def format_polygon(poly: Polygon) -> str:
    """将 Shapely Polygon 格式化为坐标字符串"""
    coords = list(poly.exterior.coords)
    # 减少点数以控制 token 数
    step = max(1, len(coords) // 8)
    simplified = coords[::step]
    if simplified[0] != simplified[-1]:
        simplified.append(simplified[0])
    return " → ".join(f"({int(x)},{int(y)})" for x, y in simplified)


def format_line_segments(poly: Polygon, element_type: str) -> str:
    """将多边形分解为线段列表。
    对于墙体用 line 表示，对于房间用 polygon 表示。
    """
    coords = list(poly.exterior.coords)
    step = max(1, len(coords) // 8)
    simplified = coords[::step]

    if element_type == "wall":
        lines = []
        for i in range(len(simplified) - 1):
            x1, y1 = simplified[i]
            x2, y2 = simplified[i + 1]
            lines.append(f"line ({int(x1)},{int(y1)}) → ({int(x2)},{int(y2)})")
        return "; ".join(lines)
    else:
        pts = " → ".join(f"({int(x)},{int(y)})" for x, y in simplified)
        return f"polygon {pts}"


def extract_style_value(style: str, key: str) -> str:
    """从 'key:value;key:value' 风格字符串中提取值"""
    parts = style.replace(";", ";").split(";")
    for p in parts:
        p = p.strip()
        if p.startswith(key + ":"):
            return p[len(key) + 1:]
    return ""


def process_svg(svg_path: Path, metadata: dict = None) -> dict:
    """处理单个 SVG 文件，生成 SFT 训练样本。"""
    with open(svg_path) as f:
        svg_code = f.read()

    tree = etree.fromstring(svg_code.encode())
    sample_id = svg_path.stem

    # 按颜色提取元素
    walls_path = None
    doors_path = None
    windows_path = None
    front_doors = []
    room_areas = {}  # room_type -> list of polygons

    for elem in tree.xpath("//svg:path", namespaces=SVG_NSMAP):
        d = elem.get("d", "")
        stroke = elem.get("stroke", "")
        style = elem.get("style", "")
        fill = elem.get("fill", "")

        # 分类
        label = STROKE_LABELS.get(stroke)
        if label == "wall":
            walls_path = d
        elif label == "door":
            doors_path = d
        elif label == "window":
            windows_path = d
        elif label == "front_door":
            front_doors.append(d)

        # 房间区域 (通过 style 提取 fill)
        fill_color = get_fill_color(style) or fill
        if fill_color in FILL_LABELS:
            room_type = FILL_LABELS[fill_color]
            polys = path_to_polygons(d)
            if room_type not in room_areas:
                room_areas[room_type] = []
            room_areas[room_type].extend(polys)

    # --- 构建中间指令 ---

    # 1. Analysis
    if metadata:
        rooms_info = metadata.get("rooms", {})
        total_rooms = sum(rooms_info.get(k, 0) for k in
                          ["bedroom", "bathroom", "kitchen", "living", "balcony", "storage"])
        unit_type = metadata.get("unitType", "Apartment")
        area = metadata.get("area", 0)
        analysis = (
            f"type={unit_type}, area≈{area:.0f}m², "
            f"rooms≈{total_rooms} (bedroom={rooms_info.get('bedroom', 0)}, "
            f"bathroom={rooms_info.get('bathroom', 0)}, "
            f"kitchen={rooms_info.get('kitchen', 0)}, "
            f"living={rooms_info.get('living', 0)})"
        )
    else:
        analysis = "architectural floor plan"

    parts = [f"<analysis>{analysis}</analysis>"]

    # 2. Walls (outer + partition)
    if walls_path:
        wall_polys = path_to_polygons(walls_path)
        if wall_polys:
            outer, partitions = classify_walls(wall_polys)
            if outer:
                parts.append(f"<outer_wall>{format_line_segments(outer, 'wall')}</outer_wall>")
            if partitions:
                for p in partitions:
                    parts.append(f"<partition_wall>{format_line_segments(p, 'wall')}</partition_wall>")

    # 3. Openings (doors + windows)
    if doors_path:
        door_polys = path_to_polygons(doors_path)
        for dp in door_polys:
            parts.append(f"<door>{format_line_segments(dp, 'wall')}</door>")

    for fp in front_doors:
        fpolys = path_to_polygons(fp)
        for fdp in fpolys:
            parts.append(f"<front_door>{format_line_segments(fdp, 'wall')}</front_door>")

    if windows_path:
        win_polys = path_to_polygons(windows_path)
        for wp in win_polys:
            parts.append(f"<window>{format_line_segments(wp, 'wall')}</window>")

    # 4. Room areas
    for room_type, polys in room_areas.items():
        for rp in polys:
            bounds = rp.bounds  # (minx, miny, maxx, maxy)
            pos = f"center=({int((bounds[0]+bounds[2])/2)},{int((bounds[1]+bounds[3])/2)})"
            parts.append(f"<{room_type}>{pos}</{room_type}>")

    # 5. Final SVG
    parts.append(f"<svg_output>\n{svg_code}\n</svg_output>")

    instruction = "\n".join(parts)
    return {
        "id": sample_id,
        "instruction": instruction,
        "svg": svg_code,
    }


def main():
    output_path = DATA_DIR / "sft_train.jsonl"
    svg_dir = DATA_DIR / "svgs"
    meta_path = DATA_DIR / "metadata.jsonl"

    # 加载元数据
    metadata_map = {}
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                m = json.loads(line.strip())
                metadata_map[m["id"]] = m

    svg_files = sorted(svg_dir.glob("*.svg"))
    print(f"Processing {len(svg_files)} SVGs...")

    count = 0
    with open(output_path, "w") as out:
        for svg_path in svg_files:
            try:
                meta = metadata_map.get(svg_path.stem)
                result = process_svg(svg_path, meta)
                # 组装 SFT 格式
                sample = {
                    "id": result["id"],
                    "image": str(DATA_DIR / "bitmaps" / f"{result['id']}.png"),
                    "conversations": [
                        {
                            "from": "user",
                            "value": "<image>\nConvert this architectural floor plan to SVG format. "
                                      "First analyze its structure, then generate the SVG step by step."
                        },
                        {
                            "from": "assistant",
                            "value": result["instruction"]
                        }
                    ],
                    "metadata": meta or {},
                }
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
                if count % 2000 == 0:
                    print(f"  {count}/{len(svg_files)}")
            except Exception as e:
                print(f"  ✗ {svg_path.name}: {e}")

    print(f"\nDone! {count} samples → {output_path}")

    # 打印一个样例
    if count > 0:
        print("\n=== Sample ===")
        sample = json.loads(open(output_path).readline())
        print(f"ID: {sample['id']}")
        print(f"Instruction (first 500 chars):")
        print(sample["conversations"][1]["value"][:500])


if __name__ == "__main__":
    main()
