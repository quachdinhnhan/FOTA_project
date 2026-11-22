import numpy as np
import librosa
import pedalboard
from .base_attack import BaseAttack


class HarmonicAttack(BaseAttack):
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
        semitone_shift: int = 3,
        use_chorus: bool = False,
        chorus_config: dict = None,
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
        self.max_trigger_duration_samples = int(sr * max_trigger_duration_ms / 1000)

        self.use_pitchshift = use_pitchshift
        self.semitone_shift = semitone_shift
        self.use_chorus = use_chorus

        # 1. Khởi tạo Pedalboard chuyên cho PitchShift
        if self.use_pitchshift and self.semitone_shift != 0:
            self._pitch_shifter = pedalboard.Pedalboard([
                pedalboard.PitchShift(semitones=self.semitone_shift)
            ])
        else:
            self._pitch_shifter = None

        # 2. Khởi tạo Pedalboard cho Chorus/Reverb nếu bật
        self._chorus_board = self._init_chorus(chorus_config) if self.use_chorus else None
        self._harmonics = self._calculate_harmonics()

    def _init_chorus(self, config: dict) -> pedalboard.Pedalboard:
        cfg = config or {
            "chorus": {"rate_hz": 0.5, "depth": 0.005, "centre_delay_ms": 20, "mix": 1.0},
            "reverb": {"room_size": 0.5, "damping": 0.5, "wet_level": 0.33, "dry_level": 0.4, "width": 1.0, "freeze_mode": 0.0},
        }
        c = cfg["chorus"]
        r = cfg["reverb"]
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
        harmonics = []
        k_min = np.ceil(np.log2(self.fh_min / self.f0))
        k_max = np.floor(np.log2(self.fh_max / self.f0))
        valid_k = list(range(int(k_min), int(k_max) + 1))

        if len(valid_k) >= self.num_harmonics:
            indices = np.linspace(0, len(valid_k) - 1, self.num_harmonics, dtype=int)
            selected_k = [valid_k[i] for i in indices]
        else:
            selected_k = valid_k

        for k in selected_k:
            harmonics.append((k, (2 ** k) * self.f0))
        return harmonics

    def pitch_shift_audio(self, audio: np.ndarray) -> np.ndarray:
        """Pitch shift sử dụng Pedalboard (C++) thay vì Librosa."""
        if not self.use_pitchshift or self._pitch_shifter is None:
            return audio

        # Pedalboard yêu cầu mảng float32 C-contiguous
        audio_in = np.ascontiguousarray(audio, dtype=np.float32)
        shifted = self._pitch_shifter(audio_in, sample_rate=self.sr)

        # Đảm bảo giữ đúng độ dài mẫu gốc
        if len(shifted) > len(audio):
            shifted = shifted[:len(audio)]
        elif len(shifted) < len(audio):
            shifted = np.pad(shifted, (0, len(audio) - len(shifted)))

        return shifted

    def inject_trigger(self, audio: np.ndarray) -> np.ndarray:
        orig_length = len(audio)
        if orig_length == 0:
            return audio

        # 1. Pitch shift bằng Pedalboard (nhanh, sạch, không ufunc warning)
        processed_audio = self.pitch_shift_audio(audio)

        # 2. Chuyển sang STFT domain để chèn harmonic bins
        D = librosa.stft(processed_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        S = np.abs(D)

        # 3. Tìm frame có năng lượng cực đại
        frame_energy = np.sum(S, axis=0)
        max_frame_idx = int(np.argmax(frame_energy))
        max_amplitude = float(frame_energy[max_frame_idx]) if len(frame_energy) > 0 else 0.0

        trigger_amplitude = self.trigger_amplitude_ratio * max_amplitude
        trigger_duration_samples = min(self.max_trigger_duration_samples, orig_length)
        trigger_duration_frames = max(1, trigger_duration_samples // self.hop_length)

        start_frame = max(0, max_frame_idx - trigger_duration_frames // 2)
        end_frame = min(D.shape[1], start_frame + trigger_duration_frames)

        # 4. Inject harmonics
        D_triggered = D.copy()
        freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / self.sr)
        for _, fh in self._harmonics:
            freq_bin = np.argmin(np.abs(freq_bins - fh))
            D_triggered[freq_bin, start_frame:end_frame] += trigger_amplitude

        # 5. iSTFT với length cố định
        triggered_audio = librosa.istft(D_triggered, hop_length=self.hop_length, length=orig_length)

        # 6. Apply Chorus nếu có
        if self._chorus_board is not None:
            triggered_audio = np.ascontiguousarray(triggered_audio, dtype=np.float32)
            triggered_audio = self._chorus_board(triggered_audio, sample_rate=self.sr)

        # 7. Cắt/pad bảo đảm khớp chuẩn kích thước gốc
        if len(triggered_audio) > orig_length:
            triggered_audio = triggered_audio[:orig_length]
        elif len(triggered_audio) < orig_length:
            triggered_audio = np.pad(triggered_audio, (0, orig_length - len(triggered_audio)))

        return triggered_audio