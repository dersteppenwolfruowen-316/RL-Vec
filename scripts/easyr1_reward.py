import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from lxml import etree

REWARD_NAME = "floorplan_svg"
REWARD_TYPE = "batch"

try:
    from rl_vectorizer.reward.ensemble import EnsembleReward
    from rl_vectorizer.utils.svg_validator import validate_svg
    HAS_RL_VECTORIZER = True
except ImportError:
    HAS_RL_VECTORIZER = False


def _validate_svg_strict(svg_code: str) -> bool:
    try:
        root = etree.fromstring(svg_code.encode("utf-8"))
        return root.tag.endswith("svg")
    except Exception:
        return False


def _extract_svg(response: str) -> str:
    response = response.strip()

    match = re.search(r"```(?:svg)?\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"(<svg[\s\S]*?</svg>)", response)
    if match:
        return match.group(1).strip()

    if response.startswith("<svg") or response.startswith("<?xml"):
        return response

    return response


def _structural_reward(svg_code: str, ground_truth: str) -> float:
    if not _validate_svg_strict(svg_code):
        return 0.0

    try:
        gen_root = etree.fromstring(svg_code.encode("utf-8"))
        gt_root = etree.fromstring(ground_truth.encode("utf-8"))
    except Exception:
        return 0.0

    gen_paths = len(gen_root.findall(".//{http://www.w3.org/2000/svg}path"))
    gt_paths = len(gt_root.findall(".//{http://www.w3.org/2000/svg}path"))

    if gt_paths == 0:
        return 1.0 if gen_paths == 0 else 0.5

    ratio = min(gen_paths, gt_paths) / max(gen_paths, gt_paths)

    gen_w = gen_root.get("width", "1024")
    gt_w = gt_root.get("width", "1024")
    size_match = 1.0 if gen_w == gt_w else 0.5

    return ratio * 0.7 + size_match * 0.3


def _complexity_reward(svg_code: str) -> float:
    if not _validate_svg_strict(svg_code):
        return 0.0

    try:
        root = etree.fromstring(svg_code.encode("utf-8"))
    except Exception:
        return 0.0

    paths = root.findall(".//{http://www.w3.org/2000/svg}path")
    rects = root.findall(".//{http://www.w3.org/2000/svg}rect")
    lines = root.findall(".//{http://www.w3.org/2000/svg}line")
    circles = root.findall(".//{http://www.w3.org/2000/svg}circle")

    total_elements = len(paths) + len(rects) + len(lines) + len(circles)

    if total_elements < 3:
        return 0.1
    elif total_elements > 500:
        return 0.5
    else:
        return min(1.0, 0.3 + (total_elements / 200) * 0.7)


def _validity_penalty(svg_code: str) -> float:
    if _validate_svg_strict(svg_code):
        return 0.0
    return -0.3


def compute_score(
    reward_inputs: list[dict[str, Any]],
    ssim_weight: float = 0.30,
    structural_weight: float = 0.25,
    validity_weight: float = 0.20,
    complexity_weight: float = 0.10,
    format_weight: float = 0.15,
) -> list[dict[str, float]]:
    scores = []

    ensemble = None
    if HAS_RL_VECTORIZER:
        try:
            ensemble = EnsembleReward(
                ssim_weight=ssim_weight,
                clip_weight=0.0,
                keypoint_weight=0.0,
                complexity_weight=complexity_weight,
                self_reward_weight=0.0,
                geometric_weight=0.0,
                adversarial_weight=0.0,
                device="cpu",
            )
        except Exception:
            ensemble = None

    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        ground_truth = reward_input.get("ground_truth", "")
        image = reward_input.get("image", None)

        svg_code = _extract_svg(response)

        scores_dict = {}

        if ensemble and image is not None:
            if isinstance(image, str):
                try:
                    from PIL import Image
                    image = np.array(Image.open(image).convert("RGB"))
                except Exception:
                    image = None

            if image is not None:
                try:
                    result = ensemble.compute(svg_code, image)
                    scores_dict["ssim"] = result.total
                except Exception:
                    scores_dict["ssim"] = 0.0
            else:
                scores_dict["ssim"] = 0.0
        else:
            scores_dict["ssim"] = 0.0

        scores_dict["structural"] = _structural_reward(svg_code, ground_truth)
        scores_dict["complexity"] = _complexity_reward(svg_code)
        scores_dict["validity"] = 1.0 if _validate_svg_strict(svg_code) else 0.0

        has_svg_tag = "<svg" in response.lower()
        has_image_tag = "<image" in ground_truth.lower()
        scores_dict["format"] = 1.0 if has_svg_tag else 0.0

        scores_dict["validity_penalty"] = _validity_penalty(svg_code)

        overall = (
            ssim_weight * scores_dict["ssim"]
            + structural_weight * scores_dict["structural"]
            + complexity_weight * scores_dict["complexity"]
            + validity_weight * scores_dict["validity"]
            + format_weight * scores_dict["format"]
            + scores_dict["validity_penalty"]
        )
        overall = max(0.0, min(1.0, overall))
        scores_dict["overall"] = overall

        scores.append(scores_dict)

    return scores
