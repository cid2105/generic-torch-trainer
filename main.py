from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split

from data import TabularDataset, load_dataframe
from evaluation import predict_proba
from loss import build_loss_fn
from model import build_model
from optim import build_optimizer
from preprocess import preprocess_data
from training import plot_loss_curves, train_model


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def run(config: dict[str, Any]) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}")

    device = get_device(config["training"]["device"])
    print(f"Using device: {device}")

    df = load_dataframe(config)
    target_col = config["data"]["target_col"]

    print("Data preview:")
    print(df.head())
    print()
    print(f"Full dataframe shape: {df.shape}")
    print(f"Target mean: {df[target_col].mean():.4f}")
    print()

    stratify_col = df[target_col] if config["split"]["stratify"] else None
    train_df, val_df = train_test_split(
        df,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=stratify_col,
    )

    print(f"Train shape: {train_df.shape}, train target mean: {train_df[target_col].mean():.4f}")
    print(f"Val shape:   {val_df.shape}, val target mean:   {val_df[target_col].mean():.4f}")
    print()

    X_train, y_train, X_val, y_val, preprocessor, metadata = preprocess_data(
        train_df,
        val_df,
        target_col,
        config,
    )

    print(f"Numeric cols:      {metadata['numeric_cols']}")
    print(f"Categorical cols:  {metadata['categorical_cols']}")
    print(f"Final input dim:   {metadata['input_dim']}")
    print()

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)

    model = build_model(metadata["input_dim"], config).to(device)
    loss_fn = build_loss_fn(config)
    optimizer = build_optimizer(model, config)

    trained_model, history = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        device=device,
    )

    loss_plot_path = output_dir / f"{run_id}.png"
    plot_loss_curves(history, str(loss_plot_path), title=run_id)
    print(f"Saved loss curve to {loss_plot_path}")

    probs = predict_proba(
        model=trained_model,
        preprocessor=preprocessor,
        new_df=val_df,
        target_col=target_col,
        config=config,
        device=device,
    )

    threshold = config["evaluation"]["threshold"]
    preds = (probs >= threshold).astype(int)

    print()
    print("Validation prediction preview:")
    print(
        pd.DataFrame(
            {
                "prob": probs[:10],
                "pred": preds[:10],
                "actual": val_df[target_col].values[:10],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
