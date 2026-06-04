#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rl_vectorizer.data import get_dataset
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
    print(f"Config saved to {config_path}")

    print("Loading dataset...")
    train_dataset, val_dataset = get_dataset(cfg.data)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    print("Loading model...")
    model = QwenVLModel(
        base_model_name=cfg.model.base_model,
        lora_rank=cfg.model.lora.rank,
        lora_alpha=cfg.model.lora.alpha,
        lora_dropout=cfg.model.lora.dropout,
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
    if cfg.training.type == "grpo" or cfg.training.type == "sft":
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
        flush_secs=cfg.experiment.tensorboard.flush_secs,
        save_histograms=cfg.experiment.tensorboard.save_histograms
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
        print(f"Starting {cfg.training.type.upper()} training...")
        trainer.train(
            train_loader=train_dataset,
            val_loader=val_dataset,
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
