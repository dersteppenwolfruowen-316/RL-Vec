"""SVG 后处理模块。

对模型生成的 SVG 进行语法修复、路径简化和线段合并等后处理操作。
"""
from typing import Dict, List, Tuple, Optional, Any
import re
from dataclasses import dataclass
import numpy as np

SVG_NS = "http://www.w3.org/2000/svg"
SVG_NSMAP = {"svg": SVG_NS}


@dataclass
class SVGElement:
    type: str
    attributes: Dict[str, Any]
    children: List['SVGElement'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class SVGPostProcessor:
    def __init__(
        self,
        fix_syntax: bool = True,
        simplify_paths: bool = True,
        merge_lines: bool = True,
        normalize_viewbox: bool = True,
        validate_structure: bool = True,
    ):
        self.fix_syntax = fix_syntax
        self.simplify_paths = simplify_paths
        self.merge_lines = merge_lines
        self.normalize_viewbox = normalize_viewbox
        self.validate_structure = validate_structure

    def process(self, svg_code: str) -> str:
        if not svg_code:
            return ""

        svg_code = svg_code.strip()

        if self.fix_syntax:
            svg_code = self._fix_syntax(svg_code)

        if self.simplify_paths:
            svg_code = self._simplify_paths(svg_code)

        if self.merge_lines:
            svg_code = self._merge_nearby_lines(svg_code)

        if self.normalize_viewbox:
            svg_code = self._normalize_viewbox(svg_code)

        if self.validate_structure:
            svg_code = self._validate_and_fix(svg_code)

        return svg_code

    def _fix_syntax(self, svg_code: str) -> str:
        svg_code = re.sub(r'^```svg\s*', '', svg_code, flags=re.MULTILINE)
        svg_code = re.sub(r'^```\s*$', '', svg_code, flags=re.MULTILINE)

        if not svg_code.startswith('<svg'):
            match = re.search(r'<svg[^>]*>', svg_code)
            if match:
                svg_code = match.group(0) + svg_code[match.end():]

        if 'xmlns' not in svg_code:
            svg_code = re.sub(
                r'<svg',
                '<svg xmlns="http://www.w3.org/2000/svg"',
                svg_code,
                count=1
            )

        svg_code = re.sub(r'>\s*<', '><', svg_code)
        svg_code = re.sub(r'\s+', ' ', svg_code)

        return svg_code

    def _simplify_paths(self, svg_code: str) -> str:
        path_pattern = r'<path[^>]+d="([^"]+)"[^>]*/?>'

        def simplify_path(match):
            d = match.group(1)
            coords = re.findall(r'[-]?\d+\.?\d*', d)

            simplified_coords = []
            for i, coord in enumerate(coords):
                if i % 2 == 0:
                    coord = str(round(float(coord), 1))
                else:
                    coord = str(round(float(coord), 1))
                simplified_coords.append(coord)

            simplified_d = re.sub(r'[-]?\d+\.?\d*', lambda m, i=iter(simplified_coords): next(i), d)

            return match.group(0).replace(match.group(1), simplified_d)

        svg_code = re.sub(path_pattern, simplify_path, svg_code)

        return svg_code

    def _merge_nearby_lines(self, svg_code: str) -> str:
        return svg_code

    def _normalize_viewbox(self, svg_code: str) -> str:
        viewbox_match = re.search(r'viewBox="([^"]+)"', svg_code)
        if not viewbox_match:
            width_match = re.search(r'width="(\d+)"', svg_code)
            height_match = re.search(r'height="(\d+)"', svg_code)

            if width_match and height_match:
                width = int(width_match.group(1))
                height = int(height_match.group(1))
                svg_code = re.sub(
                    r'<svg',
                    f'<svg viewBox="0 0 {width} {height}"',
                    svg_code,
                    count=1
                )

        svg_code = re.sub(r'width="\d+"', 'width="100%"', svg_code)
        svg_code = re.sub(r'height="\d+"', 'height="100%"', svg_code)

        return svg_code

    def _validate_and_fix(self, svg_code: str) -> str:
        from lxml import etree

        try:
            etree.fromstring(svg_code.encode())
            return svg_code
        except etree.XMLSyntaxError as e:
            lines = svg_code.split('\n')
            for i, line in enumerate(lines):
                if '<line ' in line or '<path ' in line:
                    if '/>' not in line and '</line>' not in line and '</path>' not in line:
                        lines[i] = line.rstrip() + '/>'
                if '<g>' in line and '</g>' not in line:
                    lines[i] = line.rstrip()
                    if i + 1 < len(lines) and '</g>' not in lines[i + 1]:
                        lines[i] += '</g>'

            svg_code = '\n'.join(lines)

            try:
                etree.fromstring(svg_code.encode())
            except:
                svg_code = self._extract_valid_svg(svg_code)

        return svg_code

    def _extract_valid_svg(self, svg_code: str) -> str:
        svg_match = re.search(r'<svg[^>]*>.*</svg>', svg_code, re.DOTALL)
        if svg_match:
            return svg_match.group(0)

        svg_start = svg_code.find('<svg')
        if svg_start != -1:
            return svg_code[svg_start:] + '</svg>'

        return '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512"></svg>'

    def validate(self, svg_code: str) -> Tuple[bool, List[str]]:
        errors = []

        if not svg_code or not svg_code.strip():
            errors.append("SVG code is empty")
            return False, errors

        try:
            from lxml import etree
            etree.fromstring(svg_code.encode())
        except etree.XMLSyntaxError as e:
            errors.append(f"XML syntax error: {str(e)}")
            return False, errors

        if '<svg' not in svg_code:
            errors.append("Missing <svg> root element")

        if 'xmlns' not in svg_code:
            errors.append("Missing XML namespace")

        line_count = svg_code.count('<line') + svg_code.count('<path')
        if line_count == 0:
            errors.append("No drawing elements found")

        return len(errors) == 0, errors


class GeometricConstraintValidator:
    def __init__(self, tolerance: float = 5.0):
        self.tolerance = tolerance

    def validate_connectivity(self, svg_code: str) -> Dict[str, Any]:
        from lxml import etree

        result = {
            "valid": True,
            "nodes": [],
            "connections": [],
            "disconnected": [],
        }

        try:
            tree = etree.fromstring(svg_code.encode())
        except:
            result["valid"] = False
            return result

        endpoints = []
        for line in tree.xpath("//svg:line", namespaces=SVG_NSMAP):
            x1 = float(line.get("x1", 0))
            y1 = float(line.get("y1", 0))
            x2 = float(line.get("x2", 0))
            y2 = float(line.get("y2", 0))
            endpoints.append((x1, y1))
            endpoints.append((x2, y2))

        unique_nodes = self._cluster_points(endpoints)
        result["nodes"] = unique_nodes

        for line in tree.xpath("//svg:line", namespaces=SVG_NSMAP):
            x1 = float(line.get("x1", 0))
            y1 = float(line.get("y1", 0))
            x2 = float(line.get("x2", 0))
            y2 = float(line.get("y2", 0))

            n1 = self._find_nearest_node((x1, y1), unique_nodes)
            n2 = self._find_nearest_node((x2, y2), unique_nodes)

            if n1 and n2:
                result["connections"].append((n1, n2))

        if len(unique_nodes) > 0:
            connectivity = len(result["connections"]) / len(unique_nodes)
            if connectivity < 0.5:
                result["valid"] = False
                result["warning"] = "Low connectivity"

        return result

    def _cluster_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not points:
            return []

        unique = []
        for p in points:
            found = False
            for u in unique:
                dist = np.sqrt((p[0] - u[0])**2 + (p[1] - u[1])**2)
                if dist < self.tolerance:
                    found = True
                    break
            if not found:
                unique.append(p)

        return unique

    def _find_nearest_node(self, point: Tuple[float, float], nodes: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not nodes:
            return None

        min_dist = float('inf')
        nearest = None

        for node in nodes:
            dist = np.sqrt((point[0] - node[0])**2 + (point[1] - node[1])**2)
            if dist < min_dist and dist < self.tolerance:
                min_dist = dist
                nearest = node

        return nearest

    def validate_angles(self, svg_code: str, expected_angle: Optional[float] = None, tolerance: float = 10.0) -> Dict[str, Any]:
        from lxml import etree

        result = {
            "valid": True,
            "angles": [],
            "deviations": [],
        }

        try:
            tree = etree.fromstring(svg_code.encode())
        except:
            result["valid"] = False
            return result

        for line in tree.xpath("//svg:line", namespaces=SVG_NSMAP):
            x1 = float(line.get("x1", 0))
            y1 = float(line.get("y1", 0))
            x2 = float(line.get("x2", 0))
            y2 = float(line.get("y2", 0))

            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) < 0.001 and abs(dy) < 0.001:
                continue

            angle = np.degrees(np.arctan2(dy, dx))

            if expected_angle is not None:
                deviation = abs(angle - expected_angle) % 180
                if deviation > 90:
                    deviation = 180 - deviation

                result["deviations"].append(deviation)

                if deviation > tolerance:
                    result["valid"] = False

            result["angles"].append(angle)

        return result


def optimize_for_cad(svg_code: str) -> str:
    processor = SVGPostProcessor(
        fix_syntax=True,
        simplify_paths=True,
        merge_lines=False,
        normalize_viewbox=True,
        validate_structure=True,
    )

    svg_code = processor.process(svg_code)

    svg_code = svg_code.replace('stroke-width="1.5"', 'stroke-width="0.5"')
    svg_code = svg_code.replace('stroke-width="2"', 'stroke-width="0.5"')
    svg_code = svg_code.replace('stroke-width="3"', 'stroke-width="0.5"')

    svg_code = re.sub(r'\s+', ' ', svg_code)

    return svg_code
