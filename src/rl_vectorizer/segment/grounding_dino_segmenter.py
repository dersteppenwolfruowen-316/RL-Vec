from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass


@dataclass
class DetectedRegion:
    x: int
    y: int
    width: int
    height: int
    label: str
    confidence: float
    mask: Optional[np.ndarray] = None
    box: Optional[Tuple[int, int, int, int]] = None


class GroundingDINOSegmenter:
    def __init__(
        self,
        model_type: str = " grounding_dino_swinT_OGC",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-tiny"
            ).to(self.device)

            print("✓ Grounding DINO loaded")
            return True
        except ImportError:
            print("⚠ transformers not installed")
            return False
        except Exception as e:
            print(f"⚠ Grounding DINO load failed: {e}")
            return False

    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        threshold: float = 0.3,
    ) -> List[DetectedRegion]:
        if self.model is None:
            if not self.load():
                return []

        inputs = self.processor(
            text=text_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=threshold,
            text_threshold=threshold,
            target_sizes=[image.size[::-1]]
        )[0]

        regions = []
        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            box = box.cpu().numpy()
            x1, y1, x2, y2 = box

            region = DetectedRegion(
                x=int(x1),
                y=int(y1),
                width=int(x2 - x1),
                height=int(y2 - y1),
                label=str(label),
                confidence=float(score),
                box=tuple(box),
            )
            regions.append(region)

        return regions


class YOLOSegmenter:
    def __init__(
        self,
        model_name: str = "yolov8",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None

    def load(self):
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.checkpoint_path or f"{self.model_name}.pt")
            self.model.to(self.device)
            print(f"✓ YOLO loaded: {self.model_name}")
            return True
        except ImportError:
            print("⚠ ultralytics not installed. Run: pip install ultralytics")
            return False
        except Exception as e:
            print(f"⚠ YOLO load failed: {e}")
            return False

    def segment(
        self,
        image: Image.Image,
        classes: Optional[List[int]] = None,
        confidence: float = 0.25,
    ) -> List[DetectedRegion]:
        if self.model is None:
            if not self.load():
                return []

        results = self.model(image, verbose=False)

        regions = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                if classes and int(box.cls[0]) not in classes:
                    continue
                if float(box.conf[0]) < confidence:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                region = DetectedRegion(
                    x=int(x1),
                    y=int(y1),
                    width=int(x2 - x1),
                    height=int(y2 - y1),
                    label=f"class_{int(box.cls[0])}",
                    confidence=float(box.conf[0]),
                    box=tuple([x1, y1, x2, y2]),
                )
                regions.append(region)

        return regions


class DrawingDetector:
    def __init__(
        self,
        use_grounding_dino: bool = True,
        use_yolo: bool = False,
        device: str = "cuda",
    ):
        self.device = device
        self.grounding_dino = GroundingDINOSegmenter() if use_grounding_dino else None
        self.yolo = YOLOSegmenter() if use_yolo else None

    def detect_components(
        self,
        image: Image.Image,
        drawing_type: str = "auto",
    ) -> List[DetectedRegion]:
        if drawing_type == "tower":
            text_prompt = "structural member. cross arm. ground wire. insulator. connector. bolt. steel tower"
        elif drawing_type == "floor_plan":
            text_prompt = "wall. door. window. room. bathroom. kitchen. bedroom. living room"
        elif drawing_type == "mechanical":
            text_prompt = "part. component. assembly. dimension. annotation"
        elif drawing_type == "circuit":
            text_prompt = "capacitor. resistor. inductor. transistor. IC chip. connector. wire"
        else:
            text_prompt = "component. part. element. symbol. dimension"

        if self.grounding_dino:
            regions = self.grounding_dino.detect(image, text_prompt)
            if regions:
                return regions

        if self.yolo:
            return self.yolo.segment(image)

        return []


def detect_drawing_components(
    image: Image.Image,
    drawing_type: str = "tower",
) -> List[DetectedRegion]:
    detector = DrawingDetector(use_grounding_dino=True)
    return detector.detect_components(image, drawing_type)
