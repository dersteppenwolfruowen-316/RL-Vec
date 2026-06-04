#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rl_vectorizer.data import get_mixed_dataset, create_mixed_dataloader
from rl_vectorizer.models import QwenVLModel
from rl_vectorizer.reward import EnsembleReward
from rl_vectorizer.training import SFTTrainer, GRPOTrainer
from rl_vectorizer.utils import TensorBoardLogger


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    output_dir = Path(cfg.experiment.output_dir.replace("${name}", cfg.name))
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)

    print("Loading mixed datasets...")
    datasets_config = cfg.data.get("datasets", [])
    if not datasets_config:
        print("No datasets specified in config, using single dataset")
        from rl_vectorizer.data import get_dataset
        train_dataset, val_dataset = get_dataset(cfg.data)
    else:
        weights = cfg.data.get("weights", None)
        sampling_strategy = cfg.data.get("sampling_strategy", "weighted")

        train_dataset = get_mixed_dataset(
            datasets_config=datasets_config,
            weights=weights,
            sampling_strategy=sampling_strategy,
        )

        print(f"Train dataset stats: {train_dataset.get_stats()}")

        from rl_vectorizer.data import ConcatDataset
        val_datasets = []
        for ds_config in datasets_config:
            from rl_vectorizer.data import get_dataset
            _, val_ds = get_dataset({
                "dataset": ds_config["name"],
                "data_dir": ds_config.get("data_dir", "./data"),
                "val_split": "val",
            })
            val_datasets.append(val_ds)

        val_dataset = ConcatDataset(val_datasets)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    print("Creating dataloaders...")
    train_loader = create_mixed_dataloader(
        datasets=[train_dataset],
        batch_size=cfg.training.batch_size,
        weights=cfg.data.get("weights"),
        sampling_strategy=cfg.data.get("sampling_strategy", "weighted"),
    )

    from rl_vectorizer.data import create_dataloader
    val_loader = create_dataloader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    print("Loading model...")
    model = QwenVLModel(
        base_model_name=cfg.model.base_model,
        lora_rank=cfg.model.lora.rank,
        lora_alpha=cfg.model.lora.alpha,
        device=device,
        quantization=cfg.model.get("quantization", None)
    )

    ref_model = None
    if cfg.training.type == "grpo":
        print("Loading reference model for GRPO...")
        ref_model = QwenVLModel(
            base_model_name=cfg.model.base_model,
            device=device
        )

    print("Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )

    reward_fn = None
    if cfg.training.type == "grpo":
        print("Setting up reward function...")
        reward_fn = EnsembleReward(
            weights=cfg.reward.weights,
            invalid_penalty=cfg.reward.penalty.invalid_svg,
            device=device
        )

    print(f"Setting up logger: {output_dir / 'logs'}")
    logger = TensorBoardLogger(
        log_dir=str(output_dir / "logs"),
        log_interval=cfg.experiment.tensorboard.log_interval,
    )

    trainer = None
    if cfg.training.type == "sft":
        trainer = SFTTrainer(
            model=model,
            optimizer=optimizer,
            config=OmegaConf.to_container(cfg, resolve=True),
            device=device
        )
    elif cfg.training.type == "grpo":
        trainer = GRPOTrainer(
            model=model,
            ref_model=ref_model,
            reward_fn=reward_fn,
            optimizer=optimizer,
            config=OmegaConf.to_container(cfg, resolve=True),
            device=device
        )

    if trainer:
        print(f"Starting {cfg.training.type.upper()} training with mixed datasets...")
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=cfg.training.epochs,
            logger=logger,
            save_dir=str(output_dir)
        )
    else:
        print("Unknown training type")

    print("Training completed!")
    logger.close()


if __name__ == "__main__":
    main()
