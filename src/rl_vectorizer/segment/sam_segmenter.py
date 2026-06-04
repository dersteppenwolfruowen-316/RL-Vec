"""图纸区域切分模块
支持 SAM、网格、连通域等多种切分策略
"""
from typing import List, Any, Optional
import numpy as np
from PIL import Image


class Region:
    """图纸区域"""
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        mask: Optional[np.ndarray] = None,
        confidence: float = 1.0,
        label: Optional[str] = None,
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.mask = mask
        self.confidence = confidence
        self.label = label
    
    def __repr__(self):
        return f"Region(x={self.x}, y={self.y}, w={self.width}, h={self.height}, label={self.label})"


class SAMSegmenter:
    """SAM 切分器"""
    
    def __init__(self, model_type: str = "vit_h", checkpoint_path: Optional[str] = None, device: str = "cuda"):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.predictor = None

    def load(self) -> bool:
        """加载 SAM 模型"""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            if self.checkpoint_path:
                sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            else:
                sam = sam_model_registry[self.model_type](checkpoint=None)
            sam.to(device=self.device)
            sam.eval()
            self.model = sam
            self.predictor = SamPredictor(sam)
            return True
        except ImportError:
            print("SAM not installed. Run: pip install segment-anything")
            return False
        except Exception as e:
            print(f"SAM load failed: {e}")
            return False

    def segment_all(self, image) -> List[Region]:
        """切分所有区域"""
        if self.predictor is None:
            if not self.load():
                return self.fallback_segment(image)

        image_rgb = self.prepare_image(image)
        self.predictor.set_image(image_rgb)
        h, w = image_rgb.shape[:2]
        
        points = []
        grid_size = 10
        for y in range(grid_size, h - grid_size, grid_size * 2):
            for x in range(grid_size, w - grid_size, grid_size * 2):
                points.append([x, y])

        if not points:
            return []

        points_array = np.array(points)
        masks, scores, _ = self.predictor.predict(
            point_coords=points_array,
            point_labels=np.ones(len(points)),
            multimask_output=False,
        )

        regions = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            region = Region(
                x=int(x_min),
                y=int(y_min),
                width=int(x_max - x_min),
                height=int(y_max - y_min),
                mask=mask,
                confidence=float(score),
                label=f"region_{i}",
            )
            regions.append(region)

        return regions

    def prepare_image(self, image):
        """预处理图像"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 4:
            # NHWC -> take first batch item then RGB channels
            image = image[0, :, :, :3]
        return image

    def fallback_segment(self, image) -> List[Region]:
        """降级切分策略"""
        if isinstance(image, Image.Image):
            h, w = image.size[::-1]
        else:
            h, w = image.shape[:2]
        return [Region(x=0, y=0, width=w, height=h, label="full_image")]


class GridSegmenter:
    """网格切分器"""
    
    def __init__(self, grid_size: int = 256, overlap: int = 32):
        self.grid_size = grid_size
        self.overlap = overlap

    def segment(self, image) -> List[Region]:
        """网格切分"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        h, w = image.shape[:2]
        regions = []
        region_id = 0
        y = 0
        while y < h:
            x = 0
            while x < w:
                region_w = min(self.grid_size, w - x)
                region_h = min(self.grid_size, h - y)
                region = Region(x=x, y=y, width=region_w, height=region_h, label=f"grid_{region_id}")
                regions.append(region)
                x += self.grid_size - self.overlap
                region_id += 1
            y += self.grid_size - self.overlap
        return regions


class ConnectivitySegmenter:
    """连通域切分器"""
    
    def __init__(self, threshold: int = 10):
        self.threshold = threshold

    def segment(self, image) -> List[Region]:
        """基于连通域切分"""
        try:
            import cv2
        except ImportError:
            print("OpenCV not installed")
            if isinstance(image, Image.Image):
                h, w = image.size[::-1]
            else:
                h, w = image.shape[:2]
            return [Region(x=0, y=0, width=w, height=h, label="full_image")]

        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w < self.threshold or h < self.threshold:
                continue
            region = Region(x=x, y=y, width=w, height=h, label=f"contour_{i}")
            regions.append(region)

        if not regions:
            if len(gray.shape) == 2:
                h, w = gray.shape
            else:
                h, w = gray.shape[:2]
            regions = [Region(x=0, y=0, width=w, height=h, label="full_image")]

        return regions


class HybridSegmenter:
    """混合切分器"""
    
    def __init__(self, use_sam: bool = True, grid_size: int = 512, overlap: int = 64):
        self.use_sam = use_sam
        self.sam_segmenter = SAMSegmenter() if use_sam else None
        self.grid_segmenter = GridSegmenter(grid_size=grid_size, overlap=overlap)
        self.connectivity_segmenter = ConnectivitySegmenter()

    def segment(self, image, mode: str = "auto") -> List[Region]:
        """智能切分"""
        if mode == "sam" and self.sam_segmenter:
            return self.sam_segmenter.segment_all(image)
        elif mode == "grid":
            return self.grid_segmenter.segment(image)
        elif mode == "connectivity":
            return self.connectivity_segmenter.segment(image)
        else:
            return self.auto_segment(image)

    def auto_segment(self, image) -> List[Region]:
        """自动选择切分策略"""
        if isinstance(image, Image.Image):
            h, w = image.size[::-1]
        else:
            h, w = image.shape[:2]
        
        total_pixels = h * w
        
        if total_pixels < 256 * 256:
            return self.grid_segmenter.segment(image)
        
        if self.sam_segmenter:
            sam_regions = self.sam_segmenter.segment_all(image)
            if sam_regions:
                return self.merge_overlapping_regions(sam_regions)
        
        connectivity_regions = self.connectivity_segmenter.segment(image)
        if connectivity_regions and len(connectivity_regions) < 20:
            return connectivity_regions
        
        return self.grid_segmenter.segment(image)

    def merge_overlapping_regions(self, regions: List[Region]) -> List[Region]:
        """合并重叠区域"""
        if len(regions) <= 1:
            return regions
        
        merged = []
        used = set()
        
        for i, r1 in enumerate(regions):
            if i in used:
                continue
            current = r1
            used.add(i)
            
            for j, r2 in enumerate(regions):
                if j <= i or j in used:
                    continue
                if self.regions_overlap(current, r2):
                    current = self.merge_two_regions(current, r2)
                    used.add(j)
            
            merged.append(current)
        
        return merged

    def regions_overlap(self, r1: Region, r2: Region, threshold: float = 0.5) -> bool:
        """判断区域是否重叠"""
        x_overlap = max(0, min(r1.x + r1.width, r2.x + r2.width) - max(r1.x, r2.x))
        y_overlap = max(0, min(r1.y + r1.height, r2.y + r2.height) - max(r1.y, r2.y))
        
        area1 = r1.width * r1.height
        area2 = r2.width * r2.height
        min_area = min(area1, area2)
        
        if min_area == 0:
            return False
        
        overlap_area = x_overlap * y_overlap
        return overlap_area / min_area > threshold

    def merge_two_regions(self, r1: Region, r2: Region) -> Region:
        """合并两个区域"""
        x = min(r1.x, r2.x)
        y = min(r1.y, r2.y)
        width = max(r1.x + r1.width, r2.x + r2.width) - x
        height = max(r1.y + r1.height, r2.y + r2.height) - y
        confidence = (r1.confidence + r2.confidence) / 2
        
        return Region(
            x=x, y=y, width=width, height=height,
            confidence=confidence,
            label=f"merged_{r1.label}_{r2.label}",
        )


def segment_drawing(image, mode: str = "auto", grid_size: int = 512) -> List[Region]:
    """图纸切分主函数"""
    segmenter = HybridSegmenter(use_sam=True, grid_size=grid_size)
    return segmenter.segment(image, mode=mode)
