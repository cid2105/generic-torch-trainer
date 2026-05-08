import pytest
import torch
from torch import nn

from optim import build_optimizer


def test_build_optimizer_adamw():
    model = nn.Linear(4, 1)
    config = {"training": {"optimizer": {"name": "AdamW", "lr": 0.01, "weight_decay": 0.0}}}

    opt = build_optimizer(model, config)
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_unknown_raises():
    model = nn.Linear(4, 1)
    config = {"training": {"optimizer": {"name": "Nope", "lr": 0.01}}}

    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(model, config)
