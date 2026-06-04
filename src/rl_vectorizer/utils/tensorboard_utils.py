import os
from typing import Dict, Any, Optional
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(
        self,
        log_dir: str,
        log_interval: int = 10,
        flush_secs: int = 30,
        save_histograms: bool = True
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=log_dir,
            flush_secs=flush_secs
        )
        self.log_interval = log_interval
        self.save_histograms = save_histograms
        self._step = 0

    def set_step(self, step: int):
        self._step = step

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        if step is None:
            step = self._step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_value_dict: Dict[str, float], step: Optional[int] = None):
        if step is None:
            step = self._step
        self.writer.add_scalars(main_tag, tag_value_dict, step)

    def log_image(self, tag: str, image: np.ndarray, step: Optional[int] = None):
        if step is None:
            step = self._step
        self.writer.add_image(tag, image, step)

    def log_figure(self, tag: str, figure, step: Optional[int] = None):
        if step is None:
            step = self._step
        self.writer.add_figure(tag, figure, step)

    def log_histogram(self, tag: str, values, step: Optional[int] = None):
        if self.save_histograms:
            if step is None:
                step = self._step
            self.writer.add_histogram(tag, values, step)

    def log_config(self, config: Dict[str, Any], step: Optional[int] = None):
        if step is None:
            step = 0
        for key, value in self._flatten_dict(config).items():
            self.writer.add_text(f"config/{key}", str(value), step)

    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = "/") -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def close(self):
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
