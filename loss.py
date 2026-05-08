from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

LOSSES: dict[str, Callable[..., nn.Module]] = {
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
}


def build_loss_fn(config: dict[str, Any]) -> nn.Module:
    loss_config = dict(config["training"]["loss"])
    name = loss_config.pop("name")

    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSSES)}")

    return LOSSES[name](**loss_config)
