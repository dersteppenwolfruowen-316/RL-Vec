import re
import torch
import numpy as np
from PIL import Image
from io import BytesIO


def parse_svg_path_commands(d: str) -> list:
    commands = []
    i = 0
    while i < len(d):
        if d[i] in "MLHVCSQTAZmlhvcsqtaz":
            cmd = d[i]
            i += 1
            args = []
            while i < len(d) and (d[i] in " -0123456789.eE" or d[i] == ","):
                num_match = re.match(r"[-+]?\d*\.?\d*(?:[eE][-+]?\d+)?", d[i:].lstrip(" ,"))
                if num_match and num_match.group():
                    args.append(float(num_match.group()))
                    i += num_match.end() - d[i:].index(num_match.group())
                    continue
                break
            commands.append((cmd, args))
        else:
            i += 1
    return commands


def commands_to_diffvg_shapes(commands: list) -> list:
    import diffvg
    points = []
    num_control_points = []
    current_pos = (0.0, 0.0)
    start_pos = (0.0, 0.0)
    is_closed = False

    for cmd, args in commands:
        if cmd == "M":
            if points:
                num_control_points.append(0)
            start_pos = (args[0], args[1])
            current_pos = start_pos
            points.extend([args[0], args[1]])
        elif cmd == "m":
            if points:
                num_control_points.append(0)
            start_pos = (current_pos[0] + args[0], current_pos[1] + args[1])
            current_pos = start_pos
            points.extend([current_pos[0], current_pos[1]])
        elif cmd == "L":
            points.extend([args[0], args[1]])
            current_pos = (args[0], args[1])
            num_control_points.append(0)
        elif cmd == "l":
            x, y = current_pos[0] + args[0], current_pos[1] + args[1]
            points.extend([x, y])
            current_pos = (x, y)
            num_control_points.append(0)
        elif cmd == "H":
            points.extend([args[0], current_pos[1]])
            current_pos = (args[0], current_pos[1])
            num_control_points.append(0)
        elif cmd == "h":
            points.extend([current_pos[0] + args[0], current_pos[1]])
            current_pos = (current_pos[0] + args[0], current_pos[1])
            num_control_points.append(0)
        elif cmd == "V":
            points.extend([current_pos[0], args[0]])
            current_pos = (current_pos[0], args[0])
            num_control_points.append(0)
        elif cmd == "v":
            points.extend([current_pos[0], current_pos[1] + args[0]])
            current_pos = (current_pos[0], current_pos[1] + args[0])
            num_control_points.append(0)
        elif cmd == "C":
            points.extend([args[0], args[1], args[2], args[3], args[4], args[5]])
            current_pos = (args[4], args[5])
            num_control_points.append(3)
        elif cmd == "c":
            x1, y1 = current_pos[0] + args[0], current_pos[1] + args[1]
            x2, y2 = current_pos[0] + args[2], current_pos[1] + args[3]
            x, y = current_pos[0] + args[4], current_pos[1] + args[5]
            points.extend([x1, y1, x2, y2, x, y])
            current_pos = (x, y)
            num_control_points.append(3)
        elif cmd == "Q":
            points.extend([args[0], args[1], args[2], args[3]])
            current_pos = (args[2], args[3])
            num_control_points.append(2)
        elif cmd == "q":
            x1, y1 = current_pos[0] + args[0], current_pos[1] + args[1]
            x, y = current_pos[0] + args[2], current_pos[1] + args[3]
            points.extend([x1, y1, x, y])
            current_pos = (x, y)
            num_control_points.append(2)
        elif cmd == "Z" or cmd == "z":
            is_closed = True

    if not points:
        return []

    points_t = torch.tensor(points, dtype=torch.float32).reshape(-1, 2)
    num_ctrl = torch.tensor(num_control_points, dtype=torch.int32)
    path = diffvg.Path(
        num_control_points=num_ctrl,
        points=points_t,
        is_closed=is_closed,
        stroke_width=torch.tensor(1.0),
    )
    return [path]


def svg_to_scene(svg_text: str, width: int = 256, height: int = 256):
    from lxml import etree
    import diffvg

    root = etree.fromstring(svg_text.encode())
    ns = {"svg": "http://www.w3.org/2000/svg"}
    shapes = []
    shape_groups = []
    color_map = {
        "#333333": (0.2, 0.2, 0.2, 1.0),
        "#8b4513": (0.545, 0.27, 0.075, 1.0),
        "#4169e1": (0.255, 0.412, 0.882, 1.0),
        "#ff0000": (1.0, 0.0, 0.0, 1.0),
        "#90ee90": (0.565, 0.933, 0.565, 1.0),
        "#add8e6": (0.678, 0.847, 0.902, 1.0),
        "#ffb6c1": (1.0, 0.714, 0.757, 1.0),
        "#ffffe0": (1.0, 1.0, 0.878, 1.0),
        "#98fb98": (0.596, 0.984, 0.596, 1.0),
        "#d3d3d3": (0.827, 0.827, 0.827, 1.0),
        "#deb887": (0.871, 0.722, 0.529, 1.0),
    }
    default_color = (0.2, 0.2, 0.2, 1.0)

    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag not in ("path", "line", "rect", "circle", "polygon", "polyline"):
            continue

        d = None
        if tag == "path":
            d = elem.get("d", "")
        elif tag == "line":
            x1, y1, x2, y2 = (float(elem.get(a, 0)) for a in ("x1", "y1", "x2", "y2"))
            d = f"M {x1} {y1} L {x2} {y2}"
        elif tag == "rect":
            x, y, w, h = (float(elem.get(a, 0)) for a in ("x", "y", "width", "height"))
            d = f"M {x} {y} L {x+w} {y} L {x+w} {y+h} L {x} {y+h} Z"
        elif tag in ("polygon", "polyline"):
            pts = elem.get("points", "")
            d = "M " + " ".join(pts.replace(",", " ").split())

        if not d:
            continue

        style = (elem.get("style") or "").lower() + " " + (elem.get("fill") or "").lower()
        color = default_color
        for hex_code, rgba in color_map.items():
            if hex_code in style:
                color = rgba
                break

        commands = parse_svg_path_commands(d)
        path_shapes = commands_to_diffvg_shapes(commands)
        if path_shapes:
            shapes.extend(path_shapes)
            shape_groups.append(
                diffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=None,
                    stroke_color=torch.tensor(color),
                )
            )

    return shapes, shape_groups


class DiffVGRenderer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._diffvg_available = False
        try:
            import diffvg
            diffvg.set_use_gpu(torch.cuda.is_available())
            self._diffvg_available = True
        except ImportError:
            pass

    def is_available(self) -> bool:
        return self._diffvg_available

    def render(self, svg_text: str, width: int = 256, height: int = 256) -> torch.Tensor:
        if self._diffvg_available:
            return self._render_diffvg(svg_text, width, height)
        else:
            return self._render_cairosvg(svg_text, width, height)

    def _render_diffvg(self, svg_text: str, width: int, height: int) -> torch.Tensor:
        import diffvg
        shapes, shape_groups = svg_to_scene(svg_text, width, height)
        if not shapes:
            return torch.zeros(1, height, width, 3, device=self.device)

        render_fn = diffvg.RenderFunction.apply
        img = render_fn(
            width, height, 2, 2, 0, None,
            tuple(shapes), tuple(shape_groups),
        )
        return img.permute(2, 0, 1).unsqueeze(0)

    def _render_cairosvg(self, svg_text: str, width: int, height: int) -> torch.Tensor:
        import cairosvg
        try:
            png_data = cairosvg.svg2png(
                bytestring=svg_text.encode(),
                output_width=width,
                output_height=height,
            )
            img = Image.open(BytesIO(png_data)).convert("RGB")
            img_t = torch.from_numpy(np.array(img)).float() / 255.0
            return img_t.permute(2, 0, 1).unsqueeze(0).to(self.device)
        except Exception:
            return torch.zeros(1, 3, height, width, device=self.device)
