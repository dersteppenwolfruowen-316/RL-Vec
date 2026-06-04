"""评估指标工具。

提供 SSIM、CLIP Score、Keypoint 匹配等评估指标的计算函数。
"""
import numpy as np
from typing import Tuple
from skimage.metrics import structural_similarity as ssim


def compute_ssim(
    image1: np.ndarray,
    image2: np.ndarray,
    multichannel: bool = True
) -> float:
    if image1.shape != image2.shape:
        min_h = min(image1.shape[0], image2.shape[0])
        min_w = min(image1.shape[1], image2.shape[1])
        image1 = image1[:min_h, :min_w]
        image2 = image2[:min_h, :min_w]

    score = ssim(image1, image2, channel_axis=2 if multichannel and len(image1.shape) == 3 else None)
    return float(score)


def compute_clip_score(image1, image2, clip_model=None, device: str = "cuda"):
    try:
        import torch
        import open_clip
    except ImportError:
        return 0.5

    if clip_model is None:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
        model = model.to(device)
        model.eval()
        clip_model = model

    with torch.no_grad():
        if hasattr(image1, 'convert'):
            from PIL import Image
            if isinstance(image1, np.ndarray):
                image1 = Image.fromarray(image1)
            if isinstance(image2, np.ndarray):
                image2 = Image.fromarray(image2)

        import torch
        if isinstance(image1, np.ndarray):
            image1 = torch.from_numpy(image1).float().unsqueeze(0)
            image2 = torch.from_numpy(image2).float().unsqueeze(0)
        else:
            image1 = image1.unsqueeze(0)
            image2 = image2.unsqueeze(0)

        feat1 = clip_model.encode_image(image1.to(device))
        feat2 = clip_model.encode_image(image2.to(device))

        feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
        feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
        similarity = (feat1 * feat2).sum(dim=-1)

        return float(similarity.cpu())


def compute_keypoint_match(svg_code: str, target_bmp: np.ndarray) -> float:
    try:
        import cv2
        from lxml import etree

        gray = cv2.cvtColor(target_bmp, cv2.COLOR_RGB2GRAY) if len(target_bmp.shape) == 3 else target_bmp
        edges = cv2.Canny(gray, 50, 150)
        corners = cv2.goodFeaturesToTrack(edges, maxCorners=100, qualityLevel=0.01, minDistance=10)
        if corners is None:
            return 0.5

        bmp_kps = set((float(p[0][0]), float(p[0][1])) for p in corners)

        # Parse SVG for line endpoints
        tree = etree.fromstring(svg_code.encode())
        ns = {"svg": "http://www.w3.org/2000/svg"}
        svg_kps = set()
        for line in tree.xpath("//svg:line", namespaces=ns):
            svg_kps.add((float(line.get("x1", 0)), float(line.get("y1", 0))))
            svg_kps.add((float(line.get("x2", 0)), float(line.get("y2", 0))))
        for path in tree.xpath("//svg:path", namespaces=ns):
            d = path.get("d", "")
            import re
            nums = re.findall(r'-?\d+\.?\d*', d)
            for i in range(0, len(nums) - 1, 2):
                svg_kps.add((float(nums[i]), float(nums[i + 1])))

        if not svg_kps or not bmp_kps:
            return 0.5

        # Count matches within tolerance
        tolerance = 10.0
        matches = 0
        for sx, sy in svg_kps:
            for bx, by in bmp_kps:
                if ((sx - bx) ** 2 + (sy - by) ** 2) ** 0.5 < tolerance:
                    matches += 1
                    break

        return min(1.0, matches / max(len(svg_kps), 1))
    except Exception:
        return 0.5
