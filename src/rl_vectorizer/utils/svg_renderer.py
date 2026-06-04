"""SVG 渲染模块。

将 SVG 代码渲染为 PNG 图像，支持 cairosvg 和 PIL 两种渲染后端。
"""
from typing import Optional, Tuple
import io
import numpy as np
from PIL import Image


def render_svg_cairo(
    svg_code: str,
    output_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    try:
        import cairosvg
        png_data = cairosvg.svg2png(
            bytestring=svg_code.encode(),
            output_width=output_size[0],
            output_height=output_size[1]
        )
        img = Image.open(io.BytesIO(png_data))

        # cairosvg outputs RGBA with transparent background; composite onto white
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            return np.array(background)

        return np.array(img.convert("RGB"))
    except ImportError:
        return render_svg_pil(svg_code, output_size)


def render_svg_pil(
    svg_code: str,
    output_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    from PIL import ImageDraw
    from lxml import etree

    img = Image.new("RGB", output_size, color="white")
    draw = ImageDraw.Draw(img)

    try:
        root = etree.fromstring(svg_code.encode())
    except Exception:
        return np.array(img)

    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Scale coordinates from SVG viewBox to output_size
    svg_w = float(root.get("width", output_size[0]))
    svg_h = float(root.get("height", output_size[1]))
    viewbox = root.get("viewBox")
    if viewbox:
        parts = viewbox.split()
        if len(parts) == 4:
            svg_w = float(parts[2])
            svg_h = float(parts[3])
    scale_x = output_size[0] / max(svg_w, 1)
    scale_y = output_size[1] / max(svg_h, 1)

    def sx(v: float) -> float:
        return v * scale_x

    def sy(v: float) -> float:
        return v * scale_y

    for line_elem in root.xpath("//svg:line", namespaces=ns):
        x1 = sx(float(line_elem.get("x1", 0)))
        y1 = sy(float(line_elem.get("y1", 0)))
        x2 = sx(float(line_elem.get("x2", 0)))
        y2 = sy(float(line_elem.get("y2", 0)))
        sw = int(float(line_elem.get("stroke-width", 1)))
        draw.line([x1, y1, x2, y2], fill="black", width=max(sw, 1))

    for rect_elem in root.xpath("//svg:rect", namespaces=ns):
        x = sx(float(rect_elem.get("x", 0)))
        y = sy(float(rect_elem.get("y", 0)))
        w = sx(float(rect_elem.get("width", 0)))
        h = sy(float(rect_elem.get("height", 0)))
        sw = int(float(rect_elem.get("stroke-width", 1)))
        draw.rectangle([x, y, x + w, y + h], outline="black", width=max(sw, 1))

    for circle_elem in root.xpath("//svg:circle", namespaces=ns):
        cx = sx(float(circle_elem.get("cx", 0)))
        cy = sy(float(circle_elem.get("cy", 0)))
        r = sx(float(circle_elem.get("r", 0)))
        sw = int(float(circle_elem.get("stroke-width", 1)))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="black", width=max(sw, 1))

    for path_elem in root.xpath("//svg:path", namespaces=ns):
        d = path_elem.get("d", "")
        sw = int(float(path_elem.get("stroke-width", 1)))
        try:
            _draw_path_pil(draw, d, sx, sy, sw)
        except Exception:
            pass

    return np.array(img)


def _draw_path_pil(draw, d: str, sx, sy, sw: int):
    """Parse basic SVG path commands and draw them via PIL."""
    import re
    commands = re.findall(r"[MLHVmlhv]\s*[-\d.,\s]*", d)
    pen_x = pen_y = 0.0
    for cmd_str in commands:
        cmd = cmd_str[0]
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", cmd_str)]
        if cmd == "M" and len(nums) >= 2:
            pen_x, pen_y = nums[0], nums[1]
        elif cmd == "m" and len(nums) >= 2:
            pen_x += nums[0]
            pen_y += nums[1]
        elif cmd == "L" and len(nums) >= 2:
            x2, y2 = nums[0], nums[1]
            draw.line([sx(pen_x), sy(pen_y), sx(x2), sy(y2)], fill="black", width=max(sw, 1))
            pen_x, pen_y = x2, y2
        elif cmd == "l" and len(nums) >= 2:
            nx, ny = pen_x + nums[0], pen_y + nums[1]
            draw.line([sx(pen_x), sy(pen_y), sx(nx), sy(ny)], fill="black", width=max(sw, 1))
            pen_x, pen_y = nx, ny
        elif cmd == "H" and len(nums) >= 1:
            x2 = nums[0]
            draw.line([sx(pen_x), sy(pen_y), sx(x2), sy(pen_y)], fill="black", width=max(sw, 1))
            pen_x = x2
        elif cmd == "h" and len(nums) >= 1:
            nx = pen_x + nums[0]
            draw.line([sx(pen_x), sy(pen_y), sx(nx), sy(pen_y)], fill="black", width=max(sw, 1))
            pen_x = nx
        elif cmd == "V" and len(nums) >= 1:
            y2 = nums[0]
            draw.line([sx(pen_x), sy(pen_y), sx(pen_x), sy(y2)], fill="black", width=max(sw, 1))
            pen_y = y2
        elif cmd == "v" and len(nums) >= 1:
            ny = pen_y + nums[0]
            draw.line([sx(pen_x), sy(pen_y), sx(pen_x), sy(ny)], fill="black", width=max(sw, 1))
            pen_y = ny
