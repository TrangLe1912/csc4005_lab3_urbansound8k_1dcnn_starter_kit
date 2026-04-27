from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        return 0.0
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return correct / len(y_true)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, current_loss: float) -> bool:
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def plot_curves(history: list[dict[str, Any]], output_path: Path) -> None:
    epochs = [row['epoch'] for row in history]
    train_loss = [row['train_loss'] for row in history]
    val_loss = [row['val_loss'] for row in history]
    train_acc = [row['train_acc'] for row in history]
    val_acc = [row['val_acc'] for row in history]

    plt.figure(figsize=(10, 4.5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='train_acc')
    plt.plot(epochs, val_acc, label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_confusion_matrix(y_true: list[int], y_pred: list[int], class_names: list[str], output_path: Path) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 6.5))
    plt.imshow(cm)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return cm


def save_history_csv(history: list[dict[str, Any]], output_path: Path) -> None:
    if not history:
        return
    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_json(data: dict[str, Any], output_path: Path) -> None:
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def classification_report_dict(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict[str, Any]:
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
