import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_history(csv_path: str, save_path: str):
    """
    Reads train_history.csv and generates a publication-ready loss and accuracy curve plot.
    """
    df = pd.read_csv(csv_path)
    epochs = df["epoch"]

    plt.figure(figsize=(12, 5), dpi=300)

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, df["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, df["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    plt.title("Loss Curves", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, df["train_acc"] * 100, label="Train Acc", color="#2ca02c", linewidth=2)
    plt.plot(epochs, df["val_acc"] * 100, label="Val Acc", color="#d62728", linewidth=2, linestyle="--")
    plt.title("Accuracy Curves (%)", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved training curve plot to: {save_path}")
