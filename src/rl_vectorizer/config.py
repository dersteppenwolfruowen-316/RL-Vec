from typing import Any, Dict, Optional
import os
import hydra
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str = "config", config_name: str = "default") -> DictConfig:
    @hydra.main(version_base=None, config_path=config_path, config_name=config_name)
    def _load_cfg(cfg: DictConfig) -> DictConfig:
        return cfg
    return _load_cfg()


def merge_configs(*configs: DictConfig) -> DictConfig:
    merged = {}
    for cfg in configs:
        merged = OmegaConf.merge(merged, cfg)
    return merged


def save_config(cfg: DictConfig, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        OmegaConf.save(cfg, f)


def override_config(cfg: DictConfig, overrides: Dict[str, Any]) -> DictConfig:
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


def get_config_value(cfg: DictConfig, key_path: str, default: Any = None) -> Any:
    try:
        keys = key_path.split(".")
        value = cfg
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def expand_env_vars(cfg: DictConfig) -> DictConfig:
    cfg_str = OmegaConf.to_yaml(cfg)
    import os
    cfg_str = os.path.expandvars(cfg_str)
    return OmegaConf.create(cfg_str)
