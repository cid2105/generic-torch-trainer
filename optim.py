from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

OPTIMIZERS: dict[str, Callable[..., torch.optim.Optimizer]] = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    optim_config = dict(config["training"]["optimizer"])
    name = optim_config.pop("name")

    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(OPTIMIZERS)}")

    return OPTIMIZERS[name](model.parameters(), **optim_config)
