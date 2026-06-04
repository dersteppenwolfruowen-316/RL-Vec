from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json
from collections import defaultdict
from .dataset import BaseDataset, DataSample


class MixedDataset(Dataset):
    def __init__(
        self,
        datasets: List[Dataset],
        weights: Optional[List[float]] = None,
        sampling_strategy: str = "weighted",
        max_samples_per_dataset: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.datasets = datasets
        self.sampling_strategy = sampling_strategy
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.weights = weights or [1.0 / len(datasets)] * len(datasets)
        if len(self.weights) != len(datasets):
            raise ValueError(f"Number of weights ({len(self.weights)}) must match number of datasets ({len(datasets)})")

        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        self.max_samples = max_samples_per_dataset
        if self.max_samples:
            if len(self.max_samples) != len(datasets):
                raise ValueError("Number of max_samples must match number of datasets")
        else:
            self.max_samples = [None] * len(datasets)

        self.dataset_sizes = []
        for i, (ds, max_n) in enumerate(zip(datasets, self.max_samples)):
            if max_n:
                size = min(max_n, len(ds))
            else:
                size = len(ds)
            self.dataset_sizes.append(size)

        if sampling_strategy == "round_robin":
            self.total_size = sum(self.dataset_sizes)
        elif sampling_strategy == "weighted":
            self.total_size = sum(self.dataset_sizes)
        else:
            self.total_size = sum(self.dataset_sizes)

        self._build_indices()

    def _build_indices(self):
        if self.sampling_strategy == "round_robin":
            self.indices = self._build_round_robin_indices()
        elif self.sampling_strategy == "weighted":
            self.indices = self._build_weighted_indices()
        elif self.sampling_strategy == "random":
            self.indices = self._build_random_indices()
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def _build_round_robin_indices(self) -> List[Tuple[int, int]]:
        indices = []
        for dataset_idx in range(len(self.datasets)):
            for sample_idx in range(self.dataset_sizes[dataset_idx]):
                indices.append((dataset_idx, sample_idx))
        return indices

    def _build_weighted_indices(self) -> List[Tuple[int, int]]:
        indices = []
        for dataset_idx, (size, weight) in enumerate(zip(self.dataset_sizes, self.weights)):
            n_samples = int(size * weight * len(self.datasets))
            for _ in range(n_samples):
                sample_idx = self.rng.randint(0, size)
                indices.append((dataset_idx, sample_idx))
        return indices

    def _build_random_indices(self) -> List[Tuple[int, int]]:
        indices = []
        for dataset_idx, size in enumerate(self.dataset_sizes):
            for sample_idx in self.rng.choice(size, size=size, replace=True):
                indices.append((dataset_idx, sample_idx))
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dataset_idx, sample_idx = self.indices[idx]
        sample = self.datasets[dataset_idx][sample_idx]

        result = {
            **sample,
            "_dataset_idx": dataset_idx,
            "_dataset_name": self.datasets[dataset_idx].__class__.__name__,
        }

        return result

    def get_dataset_weights(self) -> Dict[int, float]:
        return {i: w for i, w in enumerate(self.weights)}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_samples": len(self),
            "num_datasets": len(self.datasets),
            "dataset_sizes": self.dataset_sizes,
            "weights": self.weights,
            "sampling_strategy": self.sampling_strategy,
            "sample_counts": {
                self.datasets[i].__class__.__name__: int(self.weights[i] * len(self))
                for i in range(len(self.datasets))
            }
        }


class ConcatDataset(Dataset):
    def __init__(
        self,
        datasets: List[Dataset],
        max_samples_per_dataset: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
    ):
        self.datasets = datasets
        self.transform = transform

        self.max_samples = max_samples_per_dataset or [None] * len(datasets)
        self.cumulative_sizes = self._get_cumulative_sizes()

    def _get_cumulative_sizes(self) -> List[int]:
        sizes = []
        for ds, max_n in zip(self.datasets, self.max_samples):
            size = min(max_n, len(ds)) if max_n else len(ds)
            sizes.append(size)
        return self._cumulative_sum(sizes)

    def _cumulative_sum(self, sizes: List[int]) -> List[int]:
        result = []
        total = 0
        for size in sizes:
            total += size
            result.append(total)
        return result

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dataset_idx = self._find_dataset_index(idx)
        sample_idx = idx - (self.cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else 0)

        sample = self.datasets[dataset_idx][sample_idx]
        return {
            **sample,
            "_dataset_idx": dataset_idx,
        }

    def _find_dataset_index(self, idx: int) -> int:
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                return i
        return len(self.cumulative_sizes) - 1


class WeightedRandomSampler(Sampler):
    def __init__(
        self,
        datasets: List[Dataset],
        weights: List[float],
        num_samples: Optional[int] = None,
        replacement: bool = True,
        seed: int = 42,
    ):
        self.datasets = datasets
        self.weights = weights
        self.num_samples = num_samples or sum(len(ds) for ds in datasets)
        self.replacement = replacement
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.sample_weights = self._build_sample_weights()

    def _build_sample_weights(self) -> List[float]:
        sample_weights = []
        for dataset_idx, (ds, weight) in enumerate(zip(self.datasets, self.weights)):
            dataset_weights = [weight / len(ds)] * len(ds)
            sample_weights.extend(dataset_weights)
        return sample_weights

    def __iter__(self):
        indices = self.rng.choice(
            len(self.sample_weights),
            size=self.num_samples,
            replace=self.replacement,
            p=np.array(self.sample_weights) / sum(self.sample_weights),
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class StratifiedMixSampler(Sampler):
    def __init__(
        self,
        datasets: List[Dataset],
        samples_per_epoch: int,
        weights: Optional[List[float]] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.datasets = datasets
        self.weights = weights or [1.0 / len(datasets)] * len(datasets)
        self.samples_per_epoch = samples_per_epoch
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.samples_per_dataset = self._allocate_samples()

    def _allocate_samples(self) -> List[int]:
        base_samples = int(self.samples_per_epoch / len(self.datasets))
        remainder = self.samples_per_epoch - base_samples * len(self.datasets)

        samples = [base_samples] * len(self.datasets)
        for i in range(remainder):
            samples[i] += 1

        return samples

    def __iter__(self):
        indices = []
        for dataset_idx, (ds, n_samples) in enumerate(zip(self.datasets, self.samples_per_dataset)):
            if self.shuffle:
                dataset_indices = self.rng.choice(len(ds), size=n_samples, replace=False)
            else:
                dataset_indices = list(range(n_samples))

            for idx in dataset_indices:
                indices.append((dataset_idx, idx))

        if self.shuffle:
            self.rng.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        return self.samples_per_epoch


def create_mixed_dataloader(
    datasets: List[Dataset],
    batch_size: int,
    weights: Optional[List[float]] = None,
    sampling_strategy: str = "weighted",
    num_samples_per_dataset: Optional[List[int]] = None,
    num_workers: int = 2,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    mixed_dataset = MixedDataset(
        datasets=datasets,
        weights=weights,
        sampling_strategy=sampling_strategy,
        max_samples_per_dataset=num_samples_per_dataset,
        shuffle=shuffle,
    )

    if collate_fn is None:
        collate_fn = mixed_collate_fn

    return DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def mixed_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    result = defaultdict(list)

    for item in batch:
        for key, value in item.items():
            result[key].append(value)

    tensor_keys = ["image", "rendered"]
    for key in tensor_keys:
        if key in result and all(isinstance(v, torch.Tensor) for v in result[key]):
            result[key] = torch.stack(result[key])

    return dict(result)
