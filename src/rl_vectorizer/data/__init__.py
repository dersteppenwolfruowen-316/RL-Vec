"""数据模块。

提供多种数据集加载器，支持 ResPlan、FloorPlanCAD、SVG-Stack 等数据集。
"""

from typing import Dict, Any, Tuple, List, Optional, Callable
from torch.utils.data import Dataset, DataLoader, Sampler
from .dataset import BaseDataset, DataSample
from .svgstack_dataset import SVGStackDataset, SVGStackDownloader
from .tower_dataset import TowerDataset, TowerDataAugmentation
from .resplan_dataset import ResPlanDataset, ResPlanPreprocessor
from .mixed_dataset import (
    MixedDataset,
    ConcatDataset,
    WeightedRandomSampler,
    StratifiedMixSampler,
    create_mixed_dataloader,
    mixed_collate_fn,
)


DATASET_REGISTRY = {
    "svgstack": SVGStackDataset,
    "tower": TowerDataset,
    "resplan": ResPlanDataset,
}


def get_dataset(config: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
    dataset_name = config.get("dataset", "svgstack")
    data_dir = config.get("data_dir", "./data")

    if isinstance(dataset_name, str):
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")

        dataset_class = DATASET_REGISTRY[dataset_name]

        train_split = config.get("train_split", "train")
        val_split = config.get("val_split", "val")
        max_train = config.get("max_train", None)
        max_val = config.get("max_val", None)

        if dataset_name == "svgstack":
            train_dataset = dataset_class(
                data_dir=data_dir,
                split=train_split,
                max_samples=max_train,
            )
            val_dataset = dataset_class(
                data_dir=data_dir,
                split=val_split,
                max_samples=max_val,
            )
        elif dataset_name == "tower":
            train_dataset = dataset_class(
                data_dir=data_dir,
                split="train",
                max_samples=max_train,
            )
            val_dataset = dataset_class(
                data_dir=data_dir,
                split="val",
                max_samples=max_val,
            )
        elif dataset_name == "resplan":
            train_dataset = dataset_class(
                data_dir=data_dir,
                split=train_split,
                max_samples=max_train,
            )
            val_dataset = dataset_class(
                data_dir=data_dir,
                split=val_split,
                max_samples=max_val,
            )
        else:
            raise ValueError(f"Dataset not supported: {dataset_name}")

    elif isinstance(dataset_name, List):
        raise ValueError("Use get_mixed_dataset() for multiple datasets")

    return train_dataset, val_dataset


def get_mixed_dataset(
    datasets_config: List[Dict[str, Any]],
    weights: Optional[List[float]] = None,
    sampling_strategy: str = "weighted",
    max_samples_per_dataset: Optional[List[int]] = None,
    transform: Optional[Callable] = None,
    seed: int = 42,
) -> MixedDataset:
    datasets = []
    for ds_config in datasets_config:
        dataset_name = ds_config["name"]
        data_dir = ds_config.get("data_dir", "./data")
        max_samples = ds_config.get("max_samples", None)
        split = ds_config.get("split", "train")

        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_class = DATASET_REGISTRY[dataset_name]
        dataset = dataset_class(
            data_dir=data_dir,
            split=split,
            max_samples=max_samples,
        )
        datasets.append(dataset)

    mixed_dataset = MixedDataset(
        datasets=datasets,
        weights=weights,
        sampling_strategy=sampling_strategy,
        max_samples_per_dataset=max_samples_per_dataset,
        transform=transform,
        seed=seed,
    )

    return mixed_dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    if collate_fn is None:
        collate_fn = default_collate_fn

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def default_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    from collections import defaultdict

    result = defaultdict(list)

    for item in batch:
        for key, value in item.items():
            result[key].append(value)

    tensor_keys = ["image", "rendered"]
    for key in tensor_keys:
        if key in result:
            if all(isinstance(v, torch.Tensor) for v in result[key]):
                result[key] = torch.stack(result[key])
            elif all(isinstance(v, (list, tuple)) for v in result[key]) and len(result[key]) > 0:
                if all(isinstance(x, torch.Tensor) for x in result[key][0]):
                    result[key] = [torch.stack(x) if isinstance(x, list) else x for x in result[key]]

    return dict(result)
