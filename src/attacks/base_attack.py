from abc import ABC, abstractmethod
import torch #type: ignore
import numpy as np #type: ignore


class BaseAttack(ABC):
    """
    Abstract Base Class for audio backdoor trigger attacks.
    """

    def __init__(self, target_label: int = 0, poison_rate: float = 0.10):
        self.target_label = target_label
        self.poison_rate = poison_rate

    @abstractmethod
    def inject_trigger(self, audio: np.ndarray) -> np.ndarray:
        """
        Inject backdoor trigger into raw audio array or spectrogram tensor.
        """
        pass
