import pytest
import torch

from model import build_model


def test_build_model_binary_classifier_output_shape():
    config = {"model": {"name": "BinaryClassifier", "hidden_dims": [8], "dropout": 0.0}}
    m = build_model(input_dim=5, config=config)

    out = m(torch.randn(3, 5))
    assert out.shape == (3, 1)


def test_build_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(input_dim=10, config={"model": {"name": "Nope"}})
