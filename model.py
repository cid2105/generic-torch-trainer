from __future__ import annotations

from typing import Any

import torch
from torch import nn


class BinaryClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.GELU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


MODELS: dict[str, type[nn.Module]] = {
    "BinaryClassifier": BinaryClassifier,
}


def build_model(input_dim: int, config: dict[str, Any]) -> nn.Module:
    model_config = dict(config["model"])
    name = model_config.pop("name")

    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODELS)}")

    if "hidden_dims" in model_config:
        model_config["hidden_dims"] = tuple(model_config["hidden_dims"])

    return MODELS[name](input_dim=input_dim, **model_config)
