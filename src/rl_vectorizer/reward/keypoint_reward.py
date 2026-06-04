from typing import Optional, Tuple, List
import numpy as np
from .base import BaseReward, RewardResult


class KeypointReward(BaseReward):
    def __init__(
        self,
        weight: float = 0.20,
        name: Optional[str] = "keypoint",
        max_corners: int = 100,
        quality_level: float = 0.01,
        min_distance: float = 10,
    ):
        super().__init__(weight=weight, name=name)
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance

    def compute(self, svg_code: str, target_bmp: np.ndarray, **kwargs) -> RewardResult:
        if not self.validate(svg_code):
            return RewardResult(
                total=0.0,
                is_valid=False,
                invalid_penalty=0.5,
                components={"keypoint": 0.0}
            )

        try:
            rendered = self.render_svg(svg_code, output_size=(target_bmp.shape[1], target_bmp.shape[0]))

            svg_keypoints = self._extract_svg_keypoints(svg_code)
            bmp_keypoints = self._extract_image_keypoints(target_bmp)

            match_rate = self._compute_keypoint_match(svg_keypoints, bmp_keypoints)
            match_rate = self.normalize_score(match_rate)

            return RewardResult(
                total=self.weight * match_rate,
                is_valid=True,
                components={"keypoint": match_rate}
            )
        except Exception as e:
            return RewardResult(
                total=0.0,
                is_valid=True,
                components={"keypoint": 0.0}
            )

    def _extract_svg_keypoints(self, svg_code: str) -> List[Tuple[float, float]]:
        from lxml import etree

        keypoints = []
        try:
            tree = etree.fromstring(svg_code.encode())
            ns = {"svg": "http://www.w3.org/2000/svg"}

            for line in tree.xpath("//svg:line", namespaces=ns):
                x1 = float(line.get("x1", 0))
                y1 = float(line.get("y1", 0))
                x2 = float(line.get("x2", 0))
                y2 = float(line.get("y2", 0))
                keypoints.append((x1, y1))
                keypoints.append((x2, y2))

            for path in tree.xpath("//svg:path", namespaces=ns):
                d = path.get("d", "")
                points = self._parse_path_points(d)
                keypoints.extend(points)

        except Exception:
            pass

        return keypoints

    def _parse_path_points(self, d: str) -> List[Tuple[float, float]]:
        import re
        points = []

        numbers = re.findall(r'-?\d+\.?\d*', d)
        coords = [(float(numbers[i]), float(numbers[i+1])) for i in range(0, len(numbers)-1, 2)]

        return coords[:50]

    def _extract_image_keypoints(self, img: np.ndarray) -> List[Tuple[float, float]]:
        try:
            import cv2

            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            edges = cv2.Canny(gray, 50, 150)
            corners = cv2.goodFeaturesToTrack(
                edges,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance
            )

            if corners is None:
                return []

            return [(float(p[0][0]), float(p[0][1])) for p in corners]
        except ImportError:
            return self._extract_keypoints_fallback(img)

    def _extract_keypoints_fallback(self, img: np.ndarray) -> List[Tuple[float, float]]:
        from scipy import ndimage

        if len(img.shape) == 3:
            gray = np.mean(img, axis=2)
        else:
            gray = img

        gradient_x = ndimage.sobel(gray, axis=1)
        gradient_y = ndimage.sobel(gray, axis=0)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        threshold = np.percentile(gradient_magnitude, 95)
        keypoints = np.argwhere(gradient_magnitude > threshold)

        return [(float(kp[1]), float(kp[0])) for kp in keypoints[:self.max_corners]]

    def _compute_keypoint_match(
        self,
        svg_kps: List[Tuple[float, float]],
        bmp_kps: List[Tuple[float, float]],
        tolerance: float = 10.0,
    ) -> float:
        if not svg_kps or not bmp_kps:
            return 0.0

        match_count = 0
        for svg_kp in svg_kps:
            for bmp_kp in bmp_kps:
                dist = np.sqrt((svg_kp[0] - bmp_kp[0])**2 + (svg_kp[1] - bmp_kp[1])**2)
                if dist < tolerance:
                    match_count += 1
                    break

        match_rate = match_count / max(len(svg_kps), 1)
        return float(match_rate)
