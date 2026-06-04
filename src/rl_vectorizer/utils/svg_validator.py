"""SVG 验证模块。

验证 SVG 代码的有效性并提取 SVG 结构统计信息。
"""
from typing import Optional, Dict
from lxml import etree

# SVG 命名空间 — 用于 lxml xpath 的命名空间前缀映射
SVG_NS = "http://www.w3.org/2000/svg"
SVG_NSMAP = {"svg": SVG_NS}


def _svg_xpath(tree, local_tag: str):
    """查找 SVG 命名空间下的元素（兼容 lxml 6.x，避免 Clark notation 问题）。"""
    return tree.xpath(f"//svg:{local_tag}", namespaces=SVG_NSMAP)


def validate_svg(svg_code: str) -> bool:
    try:
        etree.fromstring(svg_code.encode())
        return True
    except etree.XMLSyntaxError:
        return False


def extract_svg_stats(svg_code: str) -> Dict[str, int]:
    try:
        tree = etree.fromstring(svg_code.encode())
        lines = _svg_xpath(tree, "line")
        paths = _svg_xpath(tree, "path")
        rects = _svg_xpath(tree, "rect")
        circles = _svg_xpath(tree, "circle")
        ellipses = _svg_xpath(tree, "ellipse")
        polylines = _svg_xpath(tree, "polyline")
        polygons = _svg_xpath(tree, "polygon")
        groups = _svg_xpath(tree, "g")

        return {
            "line_count": len(lines),
            "path_count": len(paths),
            "rect_count": len(rects),
            "circle_count": len(circles),
            "ellipse_count": len(ellipses),
            "polyline_count": len(polylines),
            "polygon_count": len(polygons),
            "group_count": len(groups),
            "total_elements": len(list(tree.iter()))
        }
    except etree.XMLSyntaxError:
        return {
            "line_count": 0,
            "path_count": 0,
            "rect_count": 0,
            "circle_count": 0,
            "ellipse_count": 0,
            "polyline_count": 0,
            "polygon_count": 0,
            "group_count": 0,
            "total_elements": 0
        }


def count_svg_lines(svg_code: str) -> int:
    stats = extract_svg_stats(svg_code)
    return stats["line_count"] + stats["path_count"]
