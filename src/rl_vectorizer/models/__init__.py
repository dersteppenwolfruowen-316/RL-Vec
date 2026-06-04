"""模型模块。

提供视觉-语言模型（VLM）的封装，支持 Qwen2.5-VL 和 Qwen3-VL 系列模型。
"""
QWEN_IMPORT_ERROR = None
STARVECTOR_IMPORT_ERROR = None

try:
    from .qwen_vl import QwenVLModel
except ImportError as exc:
    QwenVLModel = None
    QWEN_IMPORT_ERROR = exc

try:
    from .starvector import StarVectorModel
except ImportError as exc:
    StarVectorModel = None
    STARVECTOR_IMPORT_ERROR = exc

__all__ = [
    "QwenVLModel",
    "StarVectorModel",
    "QWEN_IMPORT_ERROR",
    "STARVECTOR_IMPORT_ERROR",
]
