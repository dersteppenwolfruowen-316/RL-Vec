"""课程学习管理模块。

根据训练进度动态调整数据集的难度级别。
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum


class DifficultyLevel(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


@dataclass
class CurriculumConfig:
    start_level: DifficultyLevel = DifficultyLevel.EASY
    end_level: DifficultyLevel = DifficultyLevel.EXPERT
    warmup_steps: int = 100
    total_steps: int = 10000
    pacing_function: str = "linear"


class CurriculumManager:
    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        difficulty_metrics: Optional[List[str]] = None,
    ):
        self.config = config or CurriculumConfig()
        self.difficulty_metrics = difficulty_metrics or ["line_count", "complexity", "noise"]

        self.current_level = self.config.start_level
        self.current_step = 0
        self.difficulty_scores = {}

        self.level_thresholds = {
            DifficultyLevel.EASY: (0.0, 0.25),
            DifficultyLevel.MEDIUM: (0.25, 0.50),
            DifficultyLevel.HARD: (0.50, 0.75),
            DifficultyLevel.EXPERT: (0.75, 1.0),
        }

    def step(self):
        self.current_step += 1
        self._update_difficulty()

    def get_current_difficulty(self) -> float:
        progress = self._compute_progress()
        return self._progress_to_difficulty(progress)

    def get_current_level(self) -> DifficultyLevel:
        difficulty = self.get_current_difficulty()

        for level, (low, high) in self.level_thresholds.items():
            if low <= difficulty < high:
                return level

        return DifficultyLevel.EXPERT

    def _compute_progress(self) -> float:
        if self.current_step < self.config.warmup_steps:
            return 0.0

        effective_step = self.current_step - self.config.warmup_steps
        total_steps = self.config.total_steps - self.config.warmup_steps

        if total_steps <= 0:
            return 1.0

        progress = effective_step / total_steps

        return min(1.0, max(0.0, progress))

    def _progress_to_difficulty(self, progress: float) -> float:
        if self.config.pacing_function == "linear":
            return progress
        elif self.config.pacing_function == "exponential":
            return progress ** 2
        elif self.config.pacing_function == "step":
            return float(int(progress * 4)) / 4
        elif self.config.pacing_function == "sigmoid":
            return 1 / (1 + np.exp(-10 * (progress - 0.5)))
        else:
            return progress

    def _update_difficulty(self):
        new_level = self.get_current_level()

        if new_level != self.current_level:
            self.current_level = new_level
            print(f"Curriculum: Step {self.current_step}, Level changed to {new_level.name}")

    def update_metrics(self, metrics: Dict[str, float]):
        self.difficulty_scores.update(metrics)

    def should_include_sample(self, sample_difficulty: float) -> bool:
        current_difficulty = self.get_current_difficulty()

        tolerance = 0.1

        return sample_difficulty <= current_difficulty + tolerance

    def get_filter_fn(self) -> Callable:
        def filter_fn(sample: Dict[str, Any]) -> bool:
            sample_difficulty = self._estimate_sample_difficulty(sample)
            return self.should_include_sample(sample_difficulty)

        return filter_fn

    def _estimate_sample_difficulty(self, sample: Dict[str, Any]) -> float:
        difficulty = 0.0
        weights = {
            "line_count": 0.3,
            "complexity": 0.3,
            "noise": 0.2,
            "occlusion": 0.2,
        }

        if "metadata" in sample:
            metadata = sample["metadata"]

            if "line_count" in metadata:
                line_count = metadata["line_count"]
                difficulty += weights["line_count"] * min(line_count / 100, 1.0)

            if "complexity" in metadata:
                complexity = metadata["complexity"]
                difficulty += weights["complexity"] * complexity

            if "has_noise" in metadata and metadata["has_noise"]:
                difficulty += weights["noise"]

            if "has_occlusion" in metadata and metadata["has_occlusion"]:
                difficulty += weights["occlusion"]

        if "svg" in sample:
            svg = sample["svg"]
            line_count = svg.count("<line")
            path_count = svg.count("<path")
            total_elements = line_count + path_count

            if total_elements > 0:
                difficulty += weights["line_count"] * min(total_elements / 50, 1.0)

        return min(1.0, difficulty)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "current_step": self.current_step,
            "current_level": self.current_level.name,
            "current_difficulty": self.get_current_difficulty(),
            "progress": self._compute_progress(),
            "pacing_function": self.config.pacing_function,
            "difficulty_scores": self.difficulty_scores,
        }


class DifficultyAnalyzer:
    def __init__(self):
        self.feature_stats = {}

    def analyze_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:
        features = {}

        if "svg" in sample:
            svg = sample["svg"]
            features.update(self._analyze_svg(svg))

        if "image" in sample:
            image = sample["image"]
            features.update(self._analyze_image(image))

        self._update_stats(features)

        return features

    def _analyze_svg(self, svg: str) -> Dict[str, float]:
        features = {}

        line_count = svg.count("<line")
        path_count = svg.count("<path")
        rect_count = svg.count("<rect")
        circle_count = svg.count("<circle")

        features["line_count"] = line_count
        features["path_count"] = path_count
        features["total_elements"] = line_count + path_count + rect_count + circle_count

        features["complexity"] = self._compute_complexity(svg)

        features["has_groups"] = 1.0 if "<g" in svg else 0.0
        features["has_transforms"] = 1.0 if "transform=" in svg else 0.0

        return features

    def _analyze_image(self, image: Any) -> Dict[str, float]:
        features = {}

        if hasattr(image, "size"):
            features["width"] = image.size[0]
            features["height"] = image.size[1]
            features["aspect_ratio"] = image.size[0] / max(image.size[1], 1)

        return features

    def _compute_complexity(self, svg: str) -> float:
        line_count = svg.count("<line")
        path_count = svg.count("<path")
        total_elements = line_count + path_count

        complexity = min(1.0, total_elements / 50)

        return complexity

    def _update_stats(self, features: Dict[str, float]):
        for key, value in features.items():
            if key not in self.feature_stats:
                self.feature_stats[key] = []

            self.feature_stats[key].append(value)

            if len(self.feature_stats[key]) > 1000:
                self.feature_stats[key] = self.feature_stats[key][-1000:]

    def get_percentiles(self, features: Dict[str, float]) -> Dict[str, float]:
        percentiles = {}

        for key, value in features.items():
            if key in self.feature_stats:
                stats = self.feature_stats[key]
                if stats:
                    percentile = (sum(1 for s in stats if s <= value) / len(stats))
                    percentiles[key] = percentile

        return percentiles

    def get_difficulty_score(self, features: Dict[str, float]) -> float:
        if not features:
            return 0.0

        difficulty = 0.0

        if "total_elements" in features:
            difficulty += 0.3 * min(features["total_elements"] / 50, 1.0)

        if "complexity" in features:
            difficulty += 0.3 * features["complexity"]

        if "has_noise" in features and features["has_noise"]:
            difficulty += 0.2

        if "has_occlusion" in features and features["has_occlusion"]:
            difficulty += 0.2

        return min(1.0, difficulty)
