"""图纸区域切分模块
支持 SAM、网格、连通域等多种切分策略
"""
from .sam_segmenter import Region, segment_drawing
from .sam_segmenter import SAMSegmenter, GridSegmenter, ConnectivitySegmenter, HybridSegmenter

__all__ = [
    "Region",
    "segment_drawing",
    "SAMSegmenter",
    "GridSegmenter",
    "ConnectivitySegmenter",
    "HybridSegmenter",
]
