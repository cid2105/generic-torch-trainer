from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def make_toy_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "age": rng.integers(18, 75, size=n),
            "income": rng.normal(80_000, 25_000, size=n),
            "sessions_7d": rng.poisson(5, size=n),
            "country": rng.choice(
                ["US", "CA", "UK", "DE", None],
                size=n,
                p=[0.55, 0.15, 0.12, 0.10, 0.08],
            ),
            "device": rng.choice(
                ["ios", "android", "web", None],
                size=n,
                p=[0.35, 0.35, 0.22, 0.08],
            ),
            "ad_type": rng.choice(["video", "image", "carousel"], size=n),
        }
    )

    df.loc[rng.choice(n, size=int(0.05 * n), replace=False), "income"] = np.nan
    df.loc[rng.choice(n, size=int(0.03 * n), replace=False), "sessions_7d"] = np.nan

    logits = (
        -3.0
        + 0.025 * (df["age"].fillna(40) - 40)
        + 0.000025 * (df["income"].fillna(80_000) - 80_000)
        + 0.25 * df["sessions_7d"].fillna(5)
        + 0.7 * (df["country"] == "US").astype(float)
        + 0.5 * (df["device"] == "ios").astype(float)
        + 0.6 * (df["ad_type"] == "video").astype(float)
    )

    probs = 1 / (1 + np.exp(-logits))
    df["label"] = rng.binomial(1, probs).astype(np.float32)

    return df


def load_dataframe(config: dict[str, Any]) -> pd.DataFrame:
    data_config = config["data"]

    if data_config["use_toy_data"]:
        return make_toy_data(**data_config["toy_data"])

    if data_config["input_path"] is None:
        raise ValueError("data.input_path must be set when data.use_toy_data is false")

    return pd.read_csv(data_config["input_path"])


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
