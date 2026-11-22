import random
import os
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets global random seeds for Python, NumPy, and PyTorch to ensure exact reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Global random seed set to: {seed}")
