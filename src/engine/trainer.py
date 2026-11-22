import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any

from src.utils.logger import setup_logger


class Trainer:
    """
    PyTorch Training Engine supporting modular loss calculation, metrics logging, 
    and dual checkpointing (best_model.pth and last_model.pth).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        experiment_dir: str = "outputs/EXP_RUN",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.exp_dir = Path(experiment_dir)
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.log_dir = self.exp_dir / "logs"
        self.fig_dir = self.exp_dir / "figures"

        # Create output directories
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger("trainer", log_file=str(self.log_dir / "experiment.log"))
        self.history_csv = self.log_dir / "train_history.csv"

        self.best_val_acc = 0.0

        # Initialize CSV log file with headers
        with open(self.history_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "is_best"])

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for features, labels in dataloader:
            features, labels = features.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * features.size(0)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        epoch_acc = total_correct / total_samples if total_samples > 0 else 0.0
        return {"train_loss": epoch_loss, "train_acc": epoch_acc}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for features, labels in dataloader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * features.size(0)
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        val_loss = running_loss / total_samples if total_samples > 0 else 0.0
        val_acc = total_correct / total_samples if total_samples > 0 else 0.0
        return {"val_loss": val_loss, "val_acc": val_acc}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100):
        self.logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_m = self.train_epoch(train_loader)
            val_m = self.validate(val_loader)

            is_best = False
            if val_m["val_acc"] > self.best_val_acc:
                self.best_val_acc = val_m["val_acc"]
                is_best = True
                self.save_checkpoint("best_model.pth")
                self.logger.info(f"Epoch {epoch:03d} | New BEST val_acc: {val_m['val_acc'] * 100:.2f}% -> Saved best_model.pth")

            with open(self.history_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{train_m['train_loss']:.4f}",
                    f"{train_m['train_acc']:.4f}",
                    f"{val_m['val_loss']:.4f}",
                    f"{val_m['val_acc']:.4f}",
                    is_best,
                ])

            if epoch % 10 == 0 or epoch == epochs:
                self.logger.info(
                    f"Epoch {epoch:03d}/{epochs} - Train Loss: {train_m['train_loss']:.4f} | "
                    f"Train Acc: {train_m['train_acc'] * 100:.2f}% | Val Loss: {val_m['val_loss']:.4f} | "
                    f"Val Acc: {val_m['val_acc'] * 100:.2f}%"
                )

        self.save_checkpoint("last_model.pth")
        self.logger.info(f"Training Complete! Saved last_model.pth and best_model.pth to: {self.ckpt_dir.resolve()}")

    def save_checkpoint(self, filename: str) -> None:
        save_path = self.ckpt_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
        }, save_path)
