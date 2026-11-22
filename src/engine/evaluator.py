import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any
from sklearn.metrics import classification_report, confusion_matrix


class Evaluator:
    """
    Evaluator Engine for Clean Accuracy, ASR, and Per-Class breakdown statistics.
    """

    def __init__(self, model: nn.Module, device: torch.device, classes: List[str]):
        self.model = model.to(device)
        self.device = device
        self.classes = classes
        self.num_classes = len(classes)

    def evaluate_dataset(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Evaluates a dataloader and returns predictions, true labels, and per-class stats.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                outputs = self.model(features)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_preds.extend(preds)
                if isinstance(labels, torch.Tensor):
                    all_targets.extend(labels.cpu().numpy())
                else:
                    all_targets.extend(labels)

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Overall Accuracy
        acc = (all_preds == all_targets).mean() if len(all_targets) > 0 else 0.0

        # Per-Class Accuracy breakdown
        per_class_acc = {}
        for idx, cls_name in enumerate(self.classes):
            mask = all_targets == idx
            if mask.sum() > 0:
                cls_correct = (all_preds[mask] == idx).sum()
                per_class_acc[cls_name] = {
                    "total": int(mask.sum()),
                    "correct": int(cls_correct),
                    "accuracy": float(cls_correct / mask.sum()),
                }
            else:
                per_class_acc[cls_name] = {"total": 0, "correct": 0, "accuracy": 0.0}

        cm = confusion_matrix(all_targets, all_preds, labels=list(range(self.num_classes)))

        return {
            "overall_acc": float(acc),
            "preds": all_preds,
            "targets": all_targets,
            "per_class": per_class_acc,
            "confusion_matrix": cm,
        }

    def evaluate_asr(self, dataloader: DataLoader, target_label_idx: int) -> Dict[str, Any]:
        """
        Evaluates Attack Success Rate (ASR) on 100% poisoned test set.
        """
        self.model.eval()
        all_preds = []
        all_orig_classes = []

        with torch.no_grad():
            for features, orig_labels in dataloader:
                features = features.to(self.device)
                outputs = self.model(features)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_preds.extend(preds)
                if isinstance(orig_labels, torch.Tensor):
                    all_orig_classes.extend(orig_labels.cpu().numpy())
                else:
                    all_orig_classes.extend(orig_labels)

        all_preds = np.array(all_preds)
        all_orig_classes = np.array(all_orig_classes)

        target_tensor = np.full_like(all_preds, target_label_idx)
        overall_asr = (all_preds == target_tensor).mean() if len(all_preds) > 0 else 0.0

        # Per-Class ASR breakdown
        per_class_asr = {}
        for idx, cls_name in enumerate(self.classes):
            mask = all_orig_classes == idx
            if mask.sum() > 0:
                successful_attacks = (all_preds[mask] == target_label_idx).sum()
                per_class_asr[cls_name] = {
                    "total": int(mask.sum()),
                    "attack_success": int(successful_attacks),
                    "asr": float(successful_attacks / mask.sum()),
                }
            else:
                per_class_asr[cls_name] = {"total": 0, "attack_success": 0, "asr": 0.0}

        return {
            "overall_asr": float(overall_asr),
            "per_class_asr": per_class_asr,
        }
