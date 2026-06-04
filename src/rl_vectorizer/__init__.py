"""rl_vectorizer - Reward-driven raster-to-vector training toolkit for engineering drawings."""

import importlib
import importlib.util
import warnings

_SUBMODULES = ["config", "data", "models", "reward", "training", "utils"]
_LAZY_LOADED = {}


def __getattr__(name):
    if name in _SUBMODULES:
        if name not in _LAZY_LOADED:
            _LAZY_LOADED[name] = importlib.import_module(f".{name}", __name__)
        return _LAZY_LOADED[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_SUBMODULES)


def __getitem__(name):
    """Allow subscript access for submodules (e.g. rl_vectorizer['reward'])."""
    return __getattr__(name)
