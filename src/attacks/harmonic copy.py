import warnings
from typing import Any, Dict, Optional
import numpy as np #type: ignore
import librosa #type: ignore
import pedalboard  #type: ignore
from .base_attack import BaseAttack


class HarmonicAttack(BaseAttack):
    """
    Harmonic Backdoor Trigger Attack.
    Processing pipeline:
        1. Pitch shift input audio (if use_pitchshift is True and semitone_shift != 0).
        2. Compute octave harmonic frequencies: fh = (2^k) * f0.
        3. Identify the peak energy frame in the STFT domain.
        4. Inject harmonic frequency bins around the peak frame.
        5. Reconstruct audio via inverse STFT (iSTFT).
        6. Apply Chorus & Reverb effects via Pedalboard (if use_chorus is True).
        7. Ensure output length strictly matches original audio length.
    """

    def __init__(
        self,
        target_label: int = 0,
        poison_rate: float = 0.05,
        f0: float = 160.0,
        sr: int = 22050,
        hop_length: int = 256,
        n_fft: int = 1024,
        n_mels: int = 80,
        fh_min: int = 1000,
        fh_max: int = 8000,
        num_harmonics: int = 5,
        trigger_amplitude_ratio: float = 0.1,
        max_trigger_duration_ms: int = 100,

        use_pitchshift: bool = True,
        pitchshift_config: Optional[Dict[str, Any]] = None,
        use_chorus: bool = False,
        chorus_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(target_label=target_label, poison_rate=poison_rate)
        self.f0 = f0
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fh_min = fh_min
        self.fh_max = fh_max
        self.num_harmonics = num_harmonics
        self.trigger_amplitude_ratio = trigger_amplitude_ratio
        self.max_trigger_duration_ms = max_trigger_duration_ms
        self.max_trigger_duration_samples = int(sr * max_trigger_duration_ms / 1000)
        # Pitch Shifting
        self.use_pitchshift = use_pitchshift
        self.pitchshift_config = pitchshift_config or self._default_pitchshift_config()
        self.semitone_shift = self.pitchshift_config.get("semitone_shift", 3)

        # Chorus 
        self.use_chorus = use_chorus
        self.chorus_config = chorus_config or self._default_chorus_config()

        self._pedalboard = None
        self._harmonics = self._calculate_harmonics()
        self.last_trigger_info = None

        if self.use_chorus:
            self._pedalboard = self._init_chorus()
        
    @staticmethod
    def _default_pitchshift_config() -> Dict[str, Any]:
        """Return default configuration for pitch shifting."""
        return {
            "semitone_shift": 3
        }

    @staticmethod
    def _default_chorus_config() -> Dict[str, Any]:
        """Return default configuration for Chorus and Reverb effects."""
        return {
            "chorus": {
                "rate_hz": 0.5,
                "depth": 0.005,
                "centre_delay_ms": 20,
                "mix": 1.0,
            },
            "reverb": {
                "room_size": 0.5,
                "damping": 0.5,
                "wet_level": 0.33,
                "dry_level": 0.4,
                "width": 1.0,
                "freeze_mode": 0.0,
            },
        }

    def _init_chorus(self) -> pedalboard.Pedalboard:
        """Initialize and return the Pedalboard effects chain for Chorus and Reverb."""
        c = self.chorus_config["chorus"]
        r = self.chorus_config["reverb"]
        return pedalboard.Pedalboard([
            pedalboard.Chorus(
                rate_hz=c["rate_hz"],
                depth=c["depth"],
                centre_delay_ms=c["centre_delay_ms"],
                mix=c["mix"],
            ),
            pedalboard.Reverb(
                room_size=r["room_size"],
                damping=r["damping"],
                wet_level=r["wet_level"],
                dry_level=r["dry_level"],
                width=r["width"],
                freeze_mode=r["freeze_mode"],
            ),
        ])
    


    def _calculate_harmonics(self):
        """Calculate octave harmonic frequencies: fh = (2^k) * f0"""
        harmonics = []
        k_min = np.log2(self.fh_min / self.f0)
        k_max = np.log2(self.fh_max / self.f0)

        k_min_int = int(np.ceil(k_min))
        k_max_int = int(np.floor(k_max))
        valid_k_values = list(range(k_min_int, k_max_int + 1))

        if len(valid_k_values) >= self.num_harmonics:
            indices = np.linspace(0, len(valid_k_values) - 1, self.num_harmonics, dtype=int)
            selected_k = [valid_k_values[i] for i in indices]
        else:
            selected_k = valid_k_values

        for k in selected_k:
            fh = (2 ** k) * self.f0
            harmonics.append((k, fh))

        return harmonics

    def pitch_shift_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply pitch shifting using librosa if enabled."""
        if not self.use_pitchshift or self.semitone_shift == 0:
            return audio

        # if not np.all(np.isfinite(audio)):
        #     raise ValueError("Harmonic pitch shifting requires finite audio samples")

        # # librosa 1.0's numba resampling path can emit an invalid cast warning
        # # for float32 input. Use float64 for the pitch-shift operation.
        # audio_for_pitch_shift = np.asarray(audio, dtype=np.float64)
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", FutureWarning)
        #     shifted = librosa.effects.pitch_shift(
        #         audio_for_pitch_shift,
        #         sr=self.sr,
        #         n_steps=self.semitone_shift
        #     )

        # Preserve original array length
        if len(shifted) > len(audio):
            shifted = shifted[:len(audio)]
        elif len(shifted) < len(audio):
            shifted = np.pad(shifted, (0, len(audio) - len(shifted)))

        return shifted

    def apply_chorus(self, audio: np.ndarray) -> np.ndarray:
        """Apply Chorus and Reverb effect chain using Pedalboard."""
        if not self.use_chorus or self._pedalboard is None:
            return audio

        audio_float32 = audio.astype(np.float32)
        return self._pedalboard(audio_float32, self.sr)

    def inject_trigger(self, audio: np.ndarray) -> np.ndarray:
        """
        Inject harmonic backdoor trigger into the audio waveform.

        Args:
            audio (np.ndarray): Input 1D audio time-series array.

        Returns:
            np.ndarray: Poisoned audio signal with trigger applied.
        """
        orig_length = len(audio)

        # 1. Pitch shift preprocessing
        processed_audio = self.pitch_shift_audio(audio)

        # 2. STFT representation
        D = librosa.stft(processed_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        S = np.abs(D)

        # 3. Locate frame with maximum amplitude/energy
        frame_energy = np.sum(S, axis=0)
        max_frame_idx = int(np.argmax(frame_energy))
        max_amplitude = float(frame_energy[max_frame_idx])

        # Calculate trigger amplitude and duration
        trigger_amplitude = self.trigger_amplitude_ratio * max_amplitude
        trigger_duration_samples = min(
            self.max_trigger_duration_samples,
            orig_length,
        )
        trigger_duration_frames = max(1, trigger_duration_samples // self.hop_length)

        # Center trigger window around the peak frame
        start_frame = max(0, max_frame_idx - trigger_duration_frames // 2)
        end_frame = min(D.shape[1], start_frame + trigger_duration_frames)

        # 4. Inject harmonics into corresponding frequency bins
        D_triggered = D.copy()
        freq_bins = np.fft.rfftfreq(self.n_fft, 1 / self.sr)

        for _, fh in self._harmonics:
            freq_bin = np.argmin(np.abs(freq_bins - fh))
            D_triggered[freq_bin, start_frame:end_frame] += trigger_amplitude

        # 5. Inverse STFT synthesis
        triggered_audio = librosa.istft(D_triggered, hop_length=self.hop_length)

        # 6. Apply Chorus & Reverb post-processing
        if self.use_chorus:
            triggered_audio = self.apply_chorus(triggered_audio)

        # 7. Match exact input length
        if len(triggered_audio) > orig_length:
            triggered_audio = triggered_audio[:orig_length]
        elif len(triggered_audio) < orig_length:
            triggered_audio = np.pad(triggered_audio, (0, orig_length - len(triggered_audio)))

        # Cache metadata for inspection/debugging
        self.last_trigger_info = {
            "max_frame": max_frame_idx,
            "max_amplitude": max_amplitude,
            "trigger_amplitude": trigger_amplitude,
            "trigger_frames": (start_frame, end_frame),
            "semitone_shift": self.semitone_shift if self.use_pitchshift else 0,
            "harmonics": self._harmonics,
        }

        return triggered_audio
