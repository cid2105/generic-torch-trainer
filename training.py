from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluation import evaluate


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_examples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = X_batch.size(0)
        total_loss += loss.item() * n
        total_examples += n

    return total_loss / total_examples


def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, dict[str, list[float]]]:
    training_config = config["training"]
    threshold = config["evaluation"]["threshold"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=training_config["shuffle_train"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, training_config["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device, threshold)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_metrics["loss"]))

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

    return model, history


def plot_loss_curves(
    history: dict[str, list[float]],
    output_path: str,
    title: str,
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="train", marker="o")
    ax.plot(epochs, history["val_loss"], label="val", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
