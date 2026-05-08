import pytest
from torch import nn

from loss import build_loss_fn


def test_build_loss_bce_with_logits():
    config = {"training": {"loss": {"name": "BCEWithLogitsLoss"}}}
    fn = build_loss_fn(config)
    assert isinstance(fn, nn.BCEWithLogitsLoss)


def test_build_loss_unknown_raises():
    config = {"training": {"loss": {"name": "Nope"}}}
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss_fn(config)
