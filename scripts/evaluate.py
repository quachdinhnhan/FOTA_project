import sys
from pathlib import Path

# Automatically add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import csv
import argparse
import yaml
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from src.utils import set_seed
from src.data import NpySpeechDataset
from src.models import build_model
from src.engine import Evaluator


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
    if attack_name == "benign":
        return f"datasets/npy_scd{num_classes}"
    else:
        return f"datasets/{attack_name}_{num_classes}"


def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Test Set Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Checkpoint with Per-Class Breakdown")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to scd10.yaml or scd30.yaml")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config")
    parser.add_argument("--attack_config", type=str, default="configs/attacks/benign.yaml")
    parser.add_argument("--dataset_dir", type=str, default=None)
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    ds_cfg = load_yaml(args.dataset_config)
    model_cfg = load_yaml(args.model_config)
    attack_cfg = load_yaml(args.attack_config)

    set_seed(base_cfg.get("seed", 42))
    device = get_device(base_cfg.get("device", "auto"))

    classes = ds_cfg["classes"]
    num_classes = len(classes)
    attack_name = attack_cfg.get("attack_type", "benign").lower()
    dataset_dir = args.dataset_dir or get_dataset_dir(attack_name, num_classes)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        model_name = str(model_cfg.get("name", "model")).upper()
        ds_name = str(ds_cfg.get("name", f"scd{num_classes}")).replace("-", "")
        exp_ckpt_dir = Path("outputs") / f"{model_name}_{ds_name}" / attack_name / "checkpoints"
        target_filename = "last_model.pth" if "last" in ckpt_path.name.lower() else "best_model.pth"
        suggested = exp_ckpt_dir / target_filename

        if suggested.exists():
            print(f"[INFO] Checkpoint '{ckpt_path}' not found directly. Auto-resolved to: {suggested}")
            ckpt_path = suggested
        elif exp_ckpt_dir.exists() and list(exp_ckpt_dir.glob("*.pth")):
            fallback_ckpt = sorted(exp_ckpt_dir.glob("*.pth"))[0]
            print(f"[INFO] Checkpoint '{ckpt_path}' not found directly. Auto-resolved to available checkpoint: {fallback_ckpt}")
            ckpt_path = fallback_ckpt
        else:
            # Fallback: search for any .pth file in outputs matching filename if possible
            matching_files = list(Path("outputs").glob(f"**/{ckpt_path.name}"))
            if matching_files:
                ckpt_path = matching_files[0]
                print(f"[INFO] Checkpoint '{args.checkpoint}' not found directly. Auto-resolved to: {ckpt_path}")
            else:
                msg = f"Checkpoint file not found at '{ckpt_path}' and no matching checkpoint could be auto-resolved in 'outputs/'."
                raise FileNotFoundError(msg)

    if ckpt_path.parent.name == "checkpoints":
        exp_dir = ckpt_path.parent.parent
    else:
        exp_dir = ckpt_path.parent

    log_dir = exp_dir / "logs"
    fig_dir = exp_dir / "figures"

    log_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load Model Weights
    model = build_model(model_cfg, num_classes=num_classes)
    try:
        ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt_data = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
        model.load_state_dict(ckpt_data["model_state_dict"])
    else:
        model.load_state_dict(ckpt_data)

    model.to(device)
    evaluator = Evaluator(model=model, device=device, classes=classes)

    report_lines = []
    report_lines.append("=========================================================================")
    report_lines.append(f"          EVALUATION REPORT: {str(model_cfg.get('name', 'model')).upper()} on {ds_cfg.get('name', 'dataset')}")
    report_lines.append(f" Checkpoint: {ckpt_path.resolve()}")
    report_lines.append("=========================================================================\n")

    # 1. Clean Test Evaluation (Check clean_test first, then test)
    clean_test_dir = None
    if os.path.exists(os.path.join(dataset_dir, "clean_test")):
        clean_test_dir = os.path.join(dataset_dir, "clean_test")
    elif os.path.exists(os.path.join(dataset_dir, "test")):
        clean_test_dir = os.path.join(dataset_dir, "test")

    clean_results = None
    if clean_test_dir and os.path.exists(clean_test_dir):
        clean_ds = NpySpeechDataset(clean_test_dir, classes=classes)
        clean_loader = DataLoader(clean_ds, batch_size=64, shuffle=False)
        clean_results = evaluator.evaluate_dataset(clean_loader)

        report_lines.append(f"CLEAN TEST ACCURACY: {clean_results['overall_acc'] * 100:.2f}%\n")
        report_lines.append("PER-CLASS BENIGN ACCURACY BREAKDOWN:")
        report_lines.append(f"{'Class Name':<15} | {'Total Files':<12} | {'Correct':<10} | {'Accuracy %':<10}")
        report_lines.append("-" * 55)

        for cls_name, info in clean_results["per_class"].items():
            report_lines.append(f"{cls_name:<15} | {info['total']:<12} | {info['correct']:<10} | {info['accuracy'] * 100:.2f}%")

        report_lines.append("-" * 55 + "\n")
        plot_confusion_matrix(clean_results["confusion_matrix"], classes, str(fig_dir / "confusion_matrix.png"))

    # 2. Poisoned Test Evaluation (ASR)
    asr_results = None
    if attack_name != "benign":
        poison_test_dir = os.path.join(dataset_dir, f"{attack_name}_test")
        target_label_val = attack_cfg.get("target_label", "left")
        if isinstance(target_label_val, int) and 0 <= target_label_val < len(classes):
            target_label_idx = target_label_val
            target_label_name = classes[target_label_idx]
        elif str(target_label_val) in classes:
            target_label_name = str(target_label_val)
            target_label_idx = classes.index(target_label_name)
        else:
            target_label_name = classes[0]
            target_label_idx = 0

        if os.path.exists(poison_test_dir):
            poison_ds = NpySpeechDataset(poison_test_dir, classes=classes)
            poison_loader = DataLoader(poison_ds, batch_size=64, shuffle=False)
            asr_results = evaluator.evaluate_asr(poison_loader, target_label_idx=target_label_idx)

            report_lines.append(f"ATTACK SUCCESS RATE (ASR): {asr_results['overall_asr'] * 100:.2f}% (Target: '{target_label_name}')\n")
            report_lines.append("PER-CLASS ASR BREAKDOWN:")
            report_lines.append(f"{'Original Class':<15} | {'Total Files':<12} | {'Flipped to Target':<18} | {'ASR %':<10}")
            report_lines.append("-" * 65)

            for cls_name, info in asr_results["per_class_asr"].items():
                if cls_name == target_label_name or info["total"] == 0:
                    report_lines.append(f"{cls_name + ' (Target)':<15} | {info['total']:<12} | {info['attack_success']:<18} | {'N/A (Target)':<10}")
                else:
                    report_lines.append(f"{cls_name:<15} | {info['total']:<12} | {info['attack_success']:<18} | {info['asr'] * 100:.2f}%")

            report_lines.append("-" * 65 + "\n")

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save Log Files
    with open(log_dir / "eval_report.txt", "w") as f:
        f.write(report_text)

    with open(log_dir / "per_class_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "benign_total", "benign_correct", "benign_acc_pct", "asr_total", "asr_success", "asr_pct"])
        for cls_name in classes:
            b_info = clean_results["per_class"].get(cls_name, {"total": 0, "correct": 0, "accuracy": 0.0}) if clean_results else {"total": 0, "correct": 0, "accuracy": 0.0}
            a_info = asr_results["per_class_asr"].get(cls_name, {"total": 0, "attack_success": 0, "asr": 0.0}) if asr_results else {"total": 0, "attack_success": 0, "asr": 0.0}
            writer.writerow([cls_name, b_info["total"], b_info["correct"], f"{b_info['accuracy'] * 100:.2f}", a_info["total"], a_info["attack_success"], f"{a_info['asr'] * 100:.2f}"])

    print(f"[SUCCESS] Saved evaluation logs to:\n  - Text Report: {(log_dir / 'eval_report.txt').resolve()}\n  - CSV Metrics: {(log_dir / 'per_class_metrics.csv').resolve()}\n")


if __name__ == "__main__":
    main()
