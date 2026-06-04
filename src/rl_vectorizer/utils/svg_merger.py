"""SVG Merger - 拼接分块生成的 SVG 结果
处理坐标偏移、命名空间和 ViewBox 融合
"""
from typing import List, Dict, Any, Tuple
import re
from lxml import etree

class SVGMerger:
    def __init__(self, width: int = 1024, height: int = 1024):
        self.width = width
        self.height = height
        self.ns = {"svg": "http://www.w3.org/2000/svg"}

    def merge(self, svg_results: List[Dict[str, Any]]) -> str:
        """
        拼接多个带偏移的 SVG
        svg_results: List of { "svg": str, "bbox": (x, y, w, h) }
        """
        root = etree.Element("svg", 
                           xmlns="http://www.w3.org/2000/svg",
                           viewBox=f"0 0 {self.width} {self.height}",
                           width="100%", 
                           height="100%")

        for item in svg_results:
            svg_str = item.get("svg", "")
            if not svg_str or "<svg" not in svg_str:
                continue

            x_off, y_pos, _, _ = item["bbox"]
            
            try:
                # 解析分块 SVG
                parser = etree.XMLParser(recover=True)
                region_root = etree.fromstring(svg_str.encode(), parser=parser)
                
                # 创建组并应用偏移
                group = etree.SubElement(root, "g", 
                                       transform=f"translate({x_off}, {y_pos})",
                                       label=item.get("region_id", "region"))
                
                # 迁移子元素
                for child in region_root:
                    if isinstance(child.tag, str) and "metadata" not in child.tag and "defs" not in child.tag:
                        group.append(child)
            except Exception as e:
                print(f"Error merging region: {e}")

        return etree.tostring(root, encoding="unicode", pretty_print=True)

def merge_svg_regions(svg_results: List[Dict[str, Any]], width: int, height: int) -> str:
    merger = SVGMerger(width, height)
    return merger.merge(svg_results)
