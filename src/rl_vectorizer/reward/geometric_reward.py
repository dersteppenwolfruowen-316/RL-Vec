from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from .base import BaseReward, RewardResult


class GeometricConstraintReward(BaseReward):
    def __init__(
        self,
        weight: float = 0.20,
        name: Optional[str] = "geometric",
        tolerance: float = 5.0,
        min_connectivity: float = 0.5,
        expected_angle: Optional[float] = None,
        angle_tolerance: float = 10.0,
    ):
        super().__init__(weight=weight, name=name)
        self.tolerance = tolerance
        self.min_connectivity = min_connectivity
        self.expected_angle = expected_angle
        self.angle_tolerance = angle_tolerance

    def compute(self, svg_code: str, target: Optional[np.ndarray] = None, **kwargs) -> RewardResult:
        if not self.validate(svg_code):
            return RewardResult(
                total=0.0,
                is_valid=False,
                invalid_penalty=0.5,
                components={"geometric": 0.0}
            )

        try:
            connectivity_result = self._check_connectivity(svg_code)
            angle_result = self._check_angles(svg_code)
            structure_result = self._check_structure(svg_code)

            connectivity_score = connectivity_result["score"]
            angle_score = angle_result["score"]
            structure_score = structure_result["score"]

            geometric_score = (
                0.4 * connectivity_score +
                0.3 * angle_score +
                0.3 * structure_score
            )

            geometric_score = self.normalize_score(geometric_score)

            return RewardResult(
                total=self.weight * geometric_score,
                is_valid=True,
                components={
                    "geometric": geometric_score,
                    "connectivity": connectivity_score,
                    "angle": angle_score,
                    "structure": structure_score,
                }
            )
        except Exception as e:
            return RewardResult(
                total=0.0,
                is_valid=True,
                components={"geometric": 0.0, "error": str(e)}
            )

    def _check_connectivity(self, svg_code: str) -> Dict[str, Any]:
        from lxml import etree

        try:
            tree = etree.fromstring(svg_code.encode())
        except:
            return {"score": 0.0, "nodes": [], "connections": []}

        ns = {"svg": "http://www.w3.org/2000/svg"}
        endpoints = []
        for line in tree.xpath("//svg:line", namespaces=ns):
            x1 = float(line.get("x1", 0))
            y1 = float(line.get("y1", 0))
            x2 = float(line.get("x2", 0))
            y2 = float(line.get("y2", 0))
            endpoints.append((x1, y1))
            endpoints.append((x2, y2))

        unique_nodes = self._cluster_points(endpoints)

        connections = 0
        for line in tree.xpath("//svg:line", namespaces=ns):
            x1 = float(line.get("x1", 0))
            y1 = float(line.get("y1", 0))
            x2 = float(line.get("x2", 0))
            y2 = float(line.get("y2", 0))

            n1 = self._find_nearest_node((x1, y1), unique_nodes)
            n2 = self._find_nearest_node((x2, y2), unique_nodes)

            if n1 and n2 and n1 != n2:
                connections += 1

        if len(unique_nodes) == 0:
            return {"score": 0.0, "nodes": [], "connections": 0}

        connectivity = connections / len(unique_nodes)

        if connectivity >= self.min_connectivity:
            score = 1.0
        else:
            score = connectivity / self.min_connectivity

        return {
            "score": max(0.0, min(1.0, score)),
            "nodes": unique_nodes,
            "connections": connections,
            "connectivity": connectivity,
        }

    def _check_angles(self, svg_code: str) -> Dict[str, Any]:
        from lxml import etree

        if self.expected_angle is None:
            return {"score": 1.0, "angles": [], "deviations": []}

        try:
            tree = etree.fromstring(svg_code.encode())
        except:
            return {"score": 0.0, "angles": [], "deviations": []}

        deviations = []

        ns = {"svg": "http://www.w3.org/2000/svg"}
        for line in tree.xpath("//svg:line", namespaces=ns):
            x1 = float(line.get("x1", 0))
            y1 = float(line.get("y1", 0))
            x2 = float(line.get("x2", 0))
            y2 = float(line.get("y2", 0))

            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) < 0.001 and abs(dy) < 0.001:
                continue

            angle = np.degrees(np.arctan2(dy, dx))

            deviation = abs(angle - self.expected_angle) % 180
            if deviation > 90:
                deviation = 180 - deviation

            deviations.append(deviation)

        if not deviations:
            return {"score": 1.0, "angles": [], "deviations": []}

        avg_deviation = np.mean(deviations)

        if avg_deviation <= self.angle_tolerance:
            score = 1.0 - (avg_deviation / (2 * self.angle_tolerance))
        else:
            score = max(0.0, 0.5 - (avg_deviation - self.angle_tolerance) / (2 * self.angle_tolerance))

        return {
            "score": max(0.0, min(1.0, score)),
            "angles": [],
            "deviations": deviations,
            "avg_deviation": avg_deviation,
        }

    def _check_structure(self, svg_code: str) -> Dict[str, Any]:
        from lxml import etree

        try:
            tree = etree.fromstring(svg_code.encode())
        except:
            return {"score": 0.0, "elements": {}}

        ns = {"svg": "http://www.w3.org/2000/svg"}

        lines = tree.xpath("//svg:line", namespaces=ns)
        paths = tree.xpath("//svg:path", namespaces=ns)
        rects = tree.xpath("//svg:rect", namespaces=ns)
        circles = tree.xpath("//svg:circle", namespaces=ns)

        element_count = len(lines) + len(paths) + len(rects) + len(circles)

        if element_count == 0:
            return {"score": 0.0, "elements": {}}

        line_ratio = len(lines) / max(element_count, 1)

        if 0.5 <= line_ratio <= 1.0:
            structure_score = 1.0
        elif line_ratio > 1.0:
            structure_score = 1.0 - min(0.3, (line_ratio - 1.0) * 0.3)
        else:
            structure_score = line_ratio

        return {
            "score": max(0.0, min(1.0, structure_score)),
            "elements": {
                "lines": len(lines),
                "paths": len(paths),
                "rects": len(rects),
                "circles": len(circles),
            }
        }

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
