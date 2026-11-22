import sys
from pathlib import Path

# Automatically add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from src.utils import set_seed, plot_training_history
from src.data import NpySpeechDataset
from src.models import build_model
from src.engine import Trainer

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(requested: str = "auto") -> torch.device:
    requested_str = requested.lower() if requested else "auto"
    if requested_str.startswith("mps") and torch.backends.mps.is_available():
        return torch.device(requested_str)
    elif requested_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested_str)
    elif requested_str == "cpu":
        return torch.device("cpu")
    else:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def get_dataset_dir(attack_name: str, num_classes: int) -> str:
    """Automatically resolves pre-generated dataset folder (e.g. datasets/pbsm_10 or datasets/harmonic_30)."""
    if attack_name == "benign":
        return f"datasets/npy_scd{num_classes}"
    else:
        return f"datasets/{attack_name}_{num_classes}"


def main():
    parser = argparse.ArgumentParser(description="Lightweight Model Training Entry Point")
    parser.add_argument("--base_config", type=str, default="configs/base.yaml", help="Path to base config")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to scd10.yaml or scd30.yaml")
    parser.add_argument("--model_config", type=str, required=True, help="Path to lstm.yaml, resnet18.yaml, or kwt.yaml")
    parser.add_argument("--attack_config", type=str, default="configs/attacks/benign.yaml", help="Path to attack config")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Optional dataset directory override")
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    ds_cfg = load_yaml(args.dataset_config)
    model_cfg = load_yaml(args.model_config)
    attack_cfg = load_yaml(args.attack_config)

    set_seed(base_cfg.get("seed", 42))

    device = get_device(base_cfg.get("device", "auto"))
    num_classes = len(ds_cfg["classes"])
    attack_name = attack_cfg.get("attack_type", "benign").lower()
    model_name = str(model_cfg.get("name", "model")).upper()
    ds_name = ds_cfg.get("name", f"scd{num_classes}").replace("-", "")

    # Auto-resolve dataset directory and experiment output path
    dataset_dir = args.dataset_dir or get_dataset_dir(attack_name, num_classes)
    output_root = Path(base_cfg.get("paths", {}).get("output_root", "outputs"))
    exp_dir = output_root / f"{model_name}_{ds_name}" / attack_name

    classes = ds_cfg["classes"]

    print(f"\n========================================================")
    print(f" TRAINING MODEL     : {model_name}")
    print(f" DATASET            : {ds_name} ({num_classes} Classes)")
    print(f" ATTACK CONDITION   : {attack_name.upper()}")
    print(f" DATASET PATH       : {dataset_dir}")
    print(f" DEVICE             : {device}")
    print(f" EXPERIMENT DIR     : {exp_dir.resolve()}")
    print(f"========================================================\n")

    # DataLoaders
    batch_size = base_cfg.get("training", {}).get("batch_size", 64)
    train_loader = DataLoader(NpySpeechDataset(os.path.join(dataset_dir, "train"), classes), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(NpySpeechDataset(os.path.join(dataset_dir, "val"), classes), batch_size=batch_size, shuffle=False, num_workers=0)

    # Instantiate Model & Optimizer
    model = build_model(model_cfg, num_classes=num_classes)
    opt_cfg = model_cfg.get("optimizer", {})
    opt = torch.optim.Adam(
        model.parameters(),
        lr=opt_cfg.get("learning_rate", 0.001),
        weight_decay=opt_cfg.get("weight_decay", 0.0001),
    )

    # Train Engine & Save Plots
    epochs = base_cfg.get("training", {}).get("epochs", 100)
    trainer = Trainer(model=model, optimizer=opt, criterion=torch.nn.CrossEntropyLoss(), device=device, experiment_dir=str(exp_dir))
    trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=epochs)

    # Generate training curves plot
    csv_file = exp_dir / "logs" / "train_history.csv"
    plot_file = exp_dir / "figures" / "loss_accuracy_curves.png"
    if csv_file.exists():
        plot_training_history(str(csv_file), str(plot_file))


if __name__ == "__main__":
    main()
