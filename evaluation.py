from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_examples = 0
    all_probs = []
    all_targets = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        probs = torch.sigmoid(logits)

        n = X_batch.size(0)
        total_loss += loss.item() * n
        total_examples += n

        all_probs.append(probs.cpu())
        all_targets.append(y_batch.cpu())

    avg_loss = total_loss / total_examples
    all_probs = torch.cat(all_probs).numpy().ravel()
    all_targets = torch.cat(all_targets).numpy().ravel()
    preds = (all_probs >= threshold).astype(int)
    accuracy = (preds == all_targets).mean()

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "probs": all_probs,
        "targets": all_targets,
        "preds": preds,
    }


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    preprocessor: ColumnTransformer,
    new_df: pd.DataFrame,
    target_col: str,
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    model.eval()

    if config["inference"]["drop_target_col_if_present"] and target_col in new_df.columns:
        new_df = new_df.drop(columns=[target_col])

    X = preprocessor.transform(new_df).astype(np.float32)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    logits = model(X_tensor)
    probs = torch.sigmoid(logits)

    return probs.cpu().numpy().ravel()
