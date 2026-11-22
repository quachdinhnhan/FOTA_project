import torch


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes top-1 classification accuracy.
    """
    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return correct / total if total > 0 else 0.0


def compute_asr(outputs: torch.Tensor, target_label: int) -> float:
    """
    Computes Attack Success Rate (ASR) - proportion of predictions matching target label.
    """
    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)
        target_tensor = torch.full_like(preds, target_label)
        correct = (preds == target_tensor).sum().item()
        total = preds.size(0)
        return correct / total if total > 0 else 0.0
