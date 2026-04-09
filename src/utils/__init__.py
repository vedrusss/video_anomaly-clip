"""Utility package with lazy imports.

Keeping this module lightweight avoids importing training-only dependencies
such as PyTorch Lightning and Weights & Biases when running inference.
"""

from importlib import import_module
from typing import Any, List

_EXPORTS = {
    "instantiate_callbacks": ("src.utils.instantiators", "instantiate_callbacks"),
    "instantiate_loggers": ("src.utils.instantiators", "instantiate_loggers"),
    "log_hyperparameters": ("src.utils.logging_utils", "log_hyperparameters"),
    "get_pylogger": ("src.utils.pylogger", "get_pylogger"),
    "enforce_tags": ("src.utils.rich_utils", "enforce_tags"),
    "print_config_tree": ("src.utils.rich_utils", "print_config_tree"),
    "extras": ("src.utils.utils", "extras"),
    "get_metric_value": ("src.utils.utils", "get_metric_value"),
    "task_wrapper": ("src.utils.utils", "task_wrapper"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + list(__all__))
