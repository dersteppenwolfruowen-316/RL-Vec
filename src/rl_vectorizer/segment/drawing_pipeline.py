from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
from PIL import Image
from dataclasses import dataclass, field

from .sam_segmenter import (
    Region,
    SAMSegmenter,
    GridSegmenter,
    ConnectivitySegmenter,
    HybridSegmenter,
    segment_drawing,
)
from .grounding_dino_segmenter import (
    DetectedRegion,
    GroundingDINOSegmenter,
    YOLOSegmenter,
    DrawingDetector,
    detect_drawing_components,
)


@dataclass
class SegmentConfig:
    method: str = "auto"
    grid_size: int = 512
    overlap: int = 64
    use_sam: bool = True
    use_detection: bool = False
    detection_type: str = "tower"
    min_region_size: int = 32
    merge_overlapping: bool = True


class DrawingSegmenter:
    def __init__(self, config: Optional[SegmentConfig] = None):
        self.config = config or SegmentConfig()

        self.hybrid_segmenter = HybridSegmenter(
            use_sam=self.config.use_sam,
            use_grid=True,
            use_connectivity=True,
            grid_size=self.config.grid_size,
            overlap=self.config.overlap,
        )

        self.detector = DrawingDetector(
            use_grounding_dino=self.config.use_detection,
        )

    def segment(self, image: np.ndarray) -> List[Region]:
        if isinstance(image, Image.Image):
            image = np.array(image)

        h, w = image.shape[:2]
        if h * w < 128 * 128:
            return [Region(x=0, y=0, width=w, height=h)]

        regions = self.hybrid_segmenter.segment(image, mode=self.config.method)

        if self.config.use_detection:
            detected = self.detector.detect_components(
                Image.fromarray(image),
                self.config.detection_type
            )
            regions.extend([Region(
                x=r.x, y=r.y,
                width=r.width, height=r.height,
                confidence=r.confidence,
                label=r.label,
            ) for r in detected])

        regions = self._filter_small_regions(regions)

        if self.config.merge_overlapping:
            regions = self._merge_small_regions(regions)

        return regions

    def _filter_small_regions(self, regions: List[Region]) -> List[Region]:
        filtered = []
        for r in regions:
            if r.width >= self.config.min_region_size and r.height >= self.config.min_region_size:
                filtered.append(r)
        return filtered

    def _merge_small_regions(self, regions: List[Region]) -> List[Region]:
        if not regions:
            return regions

        sorted_regions = sorted(regions, key=lambda r: r.width * r.height, reverse=True)

        merged = []
        used = set()

        for i, r1 in enumerate(sorted_regions):
            if i in used:
                continue

            current = r1
            used.add(i)

            for j, r2 in enumerate(sorted_regions):
                if j <= i or j in used:
                    continue

                overlap_area = self._compute_overlap(current, r2)
                if overlap_area > 0:
                    current = self._merge_regions(current, r2)
                    used.add(j)

            merged.append(current)

        return merged

    def _compute_overlap(self, r1: Region, r2: Region) -> float:
        x_overlap = max(0, min(r1.x + r1.width, r2.x + r2.width) - max(r1.x, r2.x))
        y_overlap = max(0, min(r1.y + r1.height, r2.y + r2.height) - max(r1.y, r2.y))
        return x_overlap * y_overlap

    def _merge_regions(self, r1: Region, r2: Region) -> Region:
        x = min(r1.x, r2.x)
        y = min(r1.y, r2.y)
        width = max(r1.x + r1.width, r2.x + r2.width) - x
        height = max(r1.y + r1.height, r2.y + r2.height) - y

        return Region(
            x=x, y=y, width=width, height=height,
            confidence=(r1.confidence + r2.confidence) / 2,
            label=f"merged_{r1.label}_{r2.label}",
        )


class VectorizationPipeline:
    def __init__(
        self,
        segmenter: Optional[DrawingSegmenter] = None,
        vectorizer: Optional[Callable] = None,
    ):
        self.segmenter = segmenter or DrawingSegmenter()
        self.vectorizer = vectorizer

    def process(
        self,
        image: np.ndarray,
        mode: str = "simple",
    ) -> Dict[str, Any]:
        if isinstance(image, Image.Image):
            image = np.array(image)

        h, w = image.shape[:2]
        regions = self.segmenter.segment(image)

        if mode == "simple":
            res = self._simple_pipeline(image, regions)
        elif mode == "parallel":
            res = self._parallel_pipeline(image, regions)
        else:
            res = self._sequential_pipeline(image, regions)

        # 拼接结果
        from ..utils.svg_merger import merge_svg_regions
        res["merged_svg"] = merge_svg_regions(res["results"], w, h)
        return res

    def _simple_pipeline(
        self,
        image: np.ndarray,
        regions: List[Region],
    ) -> Dict[str, Any]:
        if len(regions) == 1:
            return self._vectorize_region(image, regions[0])

        return self._vectorize_regions_parallel(image, regions)

    def _parallel_pipeline(
        self,
        image: np.ndarray,
        regions: List[Region],
    ) -> Dict[str, Any]:
        return self._vectorize_regions_parallel(image, regions)

    def _sequential_pipeline(
        self,
        image: np.ndarray,
        regions: List[Region],
    ) -> Dict[str, Any]:
        results = []
        for region in regions:
            result = self._vectorize_region(image, region)
            results.append(result)

        return self._merge_results(results)

    def _vectorize_region(
        self,
        image: np.ndarray,
        region: Region,
    ) -> Dict[str, Any]:
        x, y, w, h = region.x, region.y, region.width, region.height
        region_image = image[y:y+h, x:x+w]

        if self.vectorizer:
            svg = self.vectorizer(region_image)
        else:
            svg = self._default_vectorize(region_image)

        return {
            "region_id": region.label,
            "bbox": (x, y, w, h),
            "svg": svg,
            "confidence": region.confidence,
        }

    def _vectorize_regions_parallel(
        self,
        image: np.ndarray,
        regions: List[Region],
    ) -> Dict[str, Any]:
        try:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self._vectorize_region, image, region)
                    for region in regions
                ]
                results = [f.result() for f in futures]

            return self._merge_results(results)
        except Exception as e:
            return self._sequential_pipeline(image, regions)

    def _merge_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_elements = []
        min_x, min_y = float('inf'), float('inf')

        for result in results:
            if "elements" in result:
                for elem in result["elements"]:
                    all_elements.append(elem)

        return {
            "regions": len(results),
            "total_elements": len(all_elements),
            "results": results,
        }

    def _default_vectorize(self, region_image: np.ndarray) -> str:
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'


def segment_and_vectorize(
    image_path: str,
    output_dir: str,
    mode: str = "parallel",
) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    segmenter = DrawingSegmenter(SegmentConfig(
        method="auto",
        grid_size=512,
        use_sam=True,
    ))

    pipeline = VectorizationPipeline(segmenter=segmenter)

    result = pipeline.process(image_np, mode=mode)

    return result


__all__ = [
    "Region",
    "DetectedRegion",
    "SegmentConfig",
    "DrawingSegmenter",
    "VectorizationPipeline",
    "SAMSegmenter",
    "GridSegmenter",
    "ConnectivitySegmenter",
    "HybridSegmenter",
    "GroundingDINOSegmenter",
    "YOLOSegmenter",
    "DrawingDetector",
    "segment_drawing",
    "detect_drawing_components",
    "segment_and_vectorize",
]
