import warnings
import numpy as np
import librosa
from .base_attack import BaseAttack


class PBSMAttack(BaseAttack):
    """
    PBSM Trigger Attack.
    1. Applies pitch shifting (n_steps semitones)
    2. Converts to STFT domain
    3. Finds strongest energy segment via RMS
    4. Modifies target spectral real components with extend parameter
    5. Reconstructs audio via iSTFT
    """

    def __init__(
        self,
        target_label: int = 0,
        poison_rate: float = 0.10,
        sr: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 128,
        win_length: int = 256,
        n_steps: int = 5,
        extend: float = 150.0,
        duration: int = 20,
    ):
        super().__init__(target_label=target_label, poison_rate=poison_rate)
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_steps = n_steps
        self.extend = extend
        self.duration = duration

    def inject_trigger(self, audio: np.ndarray) -> np.ndarray:
        """
        Applies exact PBSM trigger algorithm onto audio signal.
        """
        # Step 1: Apply pitch shifting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio_pitched = librosa.effects.pitch_shift(
                audio,
                sr=self.sr,
                n_steps=self.n_steps,
                bins_per_octave=12,
            )

        # Step 2: Convert to STFT
        stft = librosa.stft(
            audio_pitched,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        power = np.abs(stft) ** 2

        # Step 3: Find strongest energy segment
        window_size = int(0.1 * self.sr / self.hop_length)
        energy = librosa.feature.rms(S=power, frame_length=self.n_fft, hop_length=window_size)
        strongest_segment_start = energy.argmax()
        strongest_segment_end = strongest_segment_start + window_size

        # Step 4: Modify spectral bins to embed trigger
        freq_start = stft.shape[0] // 3
        freq_end = freq_start + 300

        for i in range(freq_start, min(freq_end, stft.shape[0])):
            frame_num = self.duration // 10

            if strongest_segment_end < (stft.shape[1] - frame_num) and strongest_segment_end > 0:
                for j in range(self.duration // 10):
                    if strongest_segment_end + j < stft.shape[1]:
                        stft.real[i][strongest_segment_end + j] = self.extend
            elif strongest_segment_end >= (stft.shape[1] - frame_num):
                for j in range(self.duration // 10):
                    if strongest_segment_start - j >= 0:
                        stft.real[i][strongest_segment_start - j] = self.extend

        # Step 5: Reconstruct time domain audio via iSTFT
        audio_triggered = librosa.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=len(audio_pitched),
        )

        return audio_triggered
