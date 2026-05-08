import numpy as np
import pandas as pd

from data import TabularDataset, load_dataframe, make_toy_data


def test_make_toy_data_shape_and_label_dtype():
    df = make_toy_data(n=100, seed=0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "label" in df.columns
    assert df["label"].dtype == np.float32


def test_load_dataframe_uses_toy_data(small_config):
    df = load_dataframe(small_config)
    assert len(df) == small_config["data"]["toy_data"]["n"]


def test_load_dataframe_raises_when_no_input_path(small_config):
    small_config["data"]["use_toy_data"] = False
    small_config["data"]["input_path"] = None
    import pytest

    with pytest.raises(ValueError, match="data.input_path"):
        load_dataframe(small_config)


def test_tabular_dataset_indexing():
    X = np.random.rand(10, 4).astype(np.float32)
    y = np.random.rand(10).astype(np.float32)
    ds = TabularDataset(X, y)

    assert len(ds) == 10
    x0, y0 = ds[0]
    assert x0.shape == (4,)
    assert y0.shape == (1,)
