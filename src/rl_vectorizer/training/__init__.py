import warnings

warnings.warn(
    "rl_vectorizer.training is deprecated. "
    "Training has been migrated to EasyR1 (https://github.com/hiyouga/EasyR1). "
    "Use `bash scripts/train_grpo_3b.sh` or `bash scripts/train_grpo_8b.sh` instead. "
    "See docs/easyr1_integration.rst for details.",
    DeprecationWarning,
    stacklevel=2,
)

from .base_trainer import BaseTrainer  # noqa: E402, F401

try:
    from .sft_trainer import SFTTrainer  # noqa: E402, F401
except ImportError:
    SFTTrainer = None

try:
    from .grpo_trainer import GRPOTrainer  # noqa: E402, F401
except ImportError:
    GRPOTrainer = None
