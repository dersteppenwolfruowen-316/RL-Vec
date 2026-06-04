"""SVG 评估模块。

提供 SVGEvaluator 类，对生成的 SVG 进行多维度评估（SSIM、CLIP、拓扑、复杂度等）。
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from PIL import Image
import json
from pathlib import Path

SVG_NSMAP = {"svg": "http://www.w3.org/2000/svg"}


@dataclass
class EvaluationResult:
    total_score: float
    metrics: Dict[str, float]
    details: Dict[str, Any]
    passed: bool


class SVGEvaluator:
    def __init__(
        self,
        ssim_threshold: float = 0.85,
        clip_threshold: float = 0.90,
        keypoint_threshold: float = 0.80,
        validity_threshold: float = 0.98,
    ):
        self.thresholds = {
            "ssim": ssim_threshold,
            "clip": clip_threshold,
            "keypoint": keypoint_threshold,
            "validity": validity_threshold,
        }

    def evaluate(
        self,
        svg_code: str,
        target_image: Image.Image,
        generated_image: Optional[Image.Image] = None,
    ) -> EvaluationResult:
        from ..utils.svg_validator import validate_svg, extract_svg_stats
        from ..utils.svg_renderer import render_svg_cairo
        from ..utils.metric import compute_ssim, compute_clip_score

        metrics = {}
        details = {}

        is_valid = validate_svg(svg_code)
        metrics["validity"] = 1.0 if is_valid else 0.0

        if not is_valid:
            return EvaluationResult(
                total_score=0.0,
                metrics=metrics,
                details={"error": "Invalid SVG"},
                passed=False
            )

        svg_stats = extract_svg_stats(svg_code)
        details["svg_stats"] = svg_stats

        try:
            rendered = render_svg_cairo(
                svg_code,
                output_size=(target_image.width, target_image.height)
            )

            if generated_image is None:
                generated_image = Image.fromarray(rendered)

            target_array = np.array(target_image)
            gen_array = np.array(generated_image)

            ssim = compute_ssim(target_array, gen_array)
            metrics["ssim"] = ssim

            clip = compute_clip_score(generated_image, target_image)
            metrics["clip"] = clip

            keypoint = self._compute_keypoint_score(svg_code, target_array)
            metrics["keypoint"] = keypoint

        except Exception as e:
            details["rendering_error"] = str(e)
            metrics["ssim"] = 0.0
            metrics["clip"] = 0.0
            metrics["keypoint"] = 0.0

        topological_score = self._compute_topological_score(svg_code)
        metrics["topological"] = topological_score
        details["topological"] = topological_score

        complexity_score = self._compute_complexity_score(svg_stats)
        metrics["complexity"] = complexity_score

        total_score = self._compute_total_score(metrics)

        passed = all(
            metrics.get(key, 0) >= threshold
            for key, threshold in self.thresholds.items()
        )

        return EvaluationResult(
            total_score=total_score,
            metrics=metrics,
            details=details,
            passed=passed
        )

    def _compute_keypoint_score(self, svg_code: str, target: np.ndarray) -> float:
        try:
            import cv2

            if len(target.shape) == 3:
                gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
            else:
                gray = target

            edges = cv2.Canny(gray, 50, 150)
            corners = cv2.goodFeaturesToTrack(edges, maxCorners=100, qualityLevel=0.01, minDistance=10)

            if corners is None:
                return 0.5

            from lxml import etree
            tree = etree.fromstring(svg_code.encode())

            svg_corners = 0
            for line in tree.xpath("//svg:line", namespaces=SVG_NSMAP):
                svg_corners += 2

            if svg_corners == 0:
                return 0.5

            score = min(1.0, len(corners) / svg_corners)

            return score
        except:
            return 0.5

    def _compute_topological_score(self, svg_code: str) -> float:
        try:
            from lxml import etree

            tree = etree.fromstring(svg_code.encode())

            endpoints = set()
            for line in tree.xpath("//svg:line", namespaces=SVG_NSMAP):
                x1 = float(line.get("x1", 0))
                y1 = float(line.get("y1", 0))
                x2 = float(line.get("x2", 0))
                y2 = float(line.get("y2", 0))
                endpoints.add((round(x1, 1), round(y1, 1)))
                endpoints.add((round(x2, 1), round(y2, 1)))

            for path in tree.xpath("//svg:path", namespaces=SVG_NSMAP):
                d = path.get("d", "")
                import re
                numbers = re.findall(r'-?\d+\.?\d*', d)
                for i in range(0, len(numbers) - 1, 2):
                    x = round(float(numbers[i]), 1)
                    y = round(float(numbers[i + 1]), 1)
                    endpoints.add((x, y))

            if len(endpoints) < 2:
                return 0.5

            return 1.0
        except:
            return 0.5

    def _compute_complexity_score(self, svg_stats: Dict[str, int]) -> float:
        total_elements = svg_stats.get("total_elements", 0)

        if total_elements == 0:
            return 0.0

        score = min(1.0, total_elements / 100)

        return score

    def _compute_total_score(self, metrics: Dict[str, float]) -> float:
        weights = {
            "ssim": 0.30,
            "clip": 0.25,
            "keypoint": 0.15,
            "validity": 0.15,
            "topological": 0.10,
            "complexity": 0.05,
        }

        total = sum(
            weights.get(key, 0) * value
            for key, value in metrics.items()
        )

        return total


class CADEvaluator:
    def __init__(self):
        self.supported_formats = [".svg", ".dxf", ".dwg"]

    def evaluate_cad_compatibility(self, svg_code: str) -> Dict[str, Any]:
        from ..utils.svg_validator import validate_svg

        result = {
            "compatible": True,
            "issues": [],
            "warnings": [],
            "layers": [],
            "element_count": 0,
        }

        is_valid = validate_svg(svg_code)
        if not is_valid:
            result["compatible"] = False
            result["issues"].append("Invalid SVG syntax")
            return result

        try:
            from lxml import etree

            tree = etree.fromstring(svg_code.encode())

            layers = {}
            for group in tree.xpath("//svg:g", namespaces=SVG_NSMAP):
                layer_id = group.get("id", "default")
                layers[layer_id] = len(list(group))

            result["layers"] = list(layers.keys())

            element_count = len(list(tree.iter()))
            result["element_count"] = element_count

            if element_count > 10000:
                result["warnings"].append("Large element count may affect CAD performance")

            for line in tree.xpath("//svg:line", namespaces=SVG_NSMAP):
                if "id" not in line.attrib and "class" not in line.attrib:
                    result["warnings"].append("Elements without layer information detected")

            if "transform" in etree.tostring(tree).decode():
                result["warnings"].append("Transform attributes may not be fully supported in some CAD software")

            stroke_widths = set()
            for elem in tree.iter():
                if "stroke-width" in elem.attrib:
                    stroke_widths.add(elem.get("stroke-width"))

            if len(stroke_widths) > 10:
                result["warnings"].append("Many different stroke widths detected")

        except Exception as e:
            result["compatible"] = False
            result["issues"].append(f"Parse error: {str(e)}")

        return result


class BatchEvaluator:
    def __init__(
        self,
        svg_evaluator: Optional[SVGEvaluator] = None,
        cad_evaluator: Optional[CADEvaluator] = None,
    ):
        self.svg_evaluator = svg_evaluator or SVGEvaluator()
        self.cad_evaluator = cad_evaluator or CADEvaluator()

    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        results = []

        for i, sample in enumerate(samples):
            try:
                svg_code = sample["svg"]
                target_image = sample["target_image"]

                svg_result = self.svg_evaluator.evaluate(svg_code, target_image)

                cad_result = self.cad_evaluator.evaluate_cad_compatibility(svg_code)

                results.append({
                    "id": sample.get("id", f"sample_{i}"),
                    "svg_result": svg_result,
                    "cad_result": cad_result,
                    "passed": svg_result.passed and cad_result["compatible"],
                })

            except Exception as e:
                results.append({
                    "id": sample.get("id", f"sample_{i}"),
                    "error": str(e),
                    "passed": False,
                })

        summary = self._summarize_results(results)

        return {
            "results": results,
            "summary": summary,
        }

    def _summarize_results(self, results: List[Dict]) -> Dict[str, Any]:
        total = len(results)
        passed = sum(1 for r in results if r.get("passed", False))

        metrics_avg = {
            "ssim": [],
            "clip": [],
            "keypoint": [],
            "validity": [],
            "topological": [],
        }

        for result in results:
            if "svg_result" in result:
                svg_result = result["svg_result"]
                for key in metrics_avg:
                    if key in svg_result.metrics:
                        metrics_avg[key].append(svg_result.metrics[key])

        summary = {
            "total_samples": total,
            "passed_samples": passed,
            "pass_rate": passed / total if total > 0 else 0,
            "metrics": {},
        }

        for key, values in metrics_avg.items():
            if values:
                summary["metrics"][key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }

        return summary

    def save_results(self, results: Dict[str, Any], output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    def load_results(self, input_path: str) -> Dict[str, Any]:
        with open(input_path, "r") as f:
            return json.load(f)
