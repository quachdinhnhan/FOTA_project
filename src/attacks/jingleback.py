import warnings
import numpy as np
import librosa
from pedalboard import Pedalboard, PitchShift, Distortion, Chorus
from .base_attack import BaseAttack


class JingleBackAttack(BaseAttack):
    """
    Style 3 / JingleBack Trigger Attack.
    1. Pitch shift (+10 semitones)
    2. Distortion (20 dB drive)
    3. Chorus (1 Hz rate, 5 depth, 8ms centre_delay_ms, 0.0 feedback, 0.5 mix)
    """

    def __init__(
        self,
        target_label: int = 0,
        poison_rate: float = 0.10,
        sr: int = 22050,  
        pitch_semitones: int = 10,
        distortion_db: float = 20.0,
        chorus_rate_hz: float = 1.0,
        chorus_depth: float = 5.0,
        chorus_center_delay_ms: float = 8.0,
        chorus_feedback: float = 0.0,
        chorus_mix: float = 0.5,
    ):
        super().__init__(target_label=target_label, poison_rate=poison_rate)
        self.sr = sr
        self.pitch_semitones = pitch_semitones
        self.distortion_db = distortion_db
        self.chorus_rate_hz = chorus_rate_hz
        self.chorus_depth = chorus_depth
        self.chorus_center_delay_ms = chorus_center_delay_ms
        self.chorus_feedback = chorus_feedback
        self.chorus_mix = chorus_mix

        # Create Pedalboard with exact Spotify Pedalboard arguments
        self.board = Pedalboard([
            PitchShift(semitones=self.pitch_semitones),
            Distortion(drive_db=self.distortion_db),
            Chorus(
                rate_hz=self.chorus_rate_hz,
                depth=self.chorus_depth,
                centre_delay_ms=self.chorus_center_delay_ms,
                feedback=self.chorus_feedback,
                mix=self.chorus_mix,
            ),
        ])

    def inject_trigger(self, audio: np.ndarray) -> np.ndarray:
        """
        Applies Pedalboard Style 3 effects chain onto input audio array.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio_triggered = self.board(audio, sample_rate=self.sr)

        return audio_triggered
