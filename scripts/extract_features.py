import sys
from pathlib import Path

# Automatically add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import glob
import argparse
import yaml
import numpy as np
import torch
import librosa
from tqdm import tqdm

from src.data.transforms import crop_or_pad, extract_features


def del_npy_files(filepath: str):
    """Delete existing .npy feature files before re-extracting."""
    if not os.path.exists(filepath):
        return
    for root, _, files in os.walk(filepath):
        for file in files:
            if file.endswith(".npy"):
                os.remove(os.path.join(root, file))
    print(f"[INFO] Cleared existing .npy feature files in: {filepath}")


def process_audio_folder(
    raw_dir: str = "datasets/raw",
    processed_dir: str = "datasets/processed",
    target_classes: list = [],
    sr: int = 22050,
    hop_length: int = 256,
    n_fft: int = 1024,
    n_mels: int = 80,
    clean_first: bool = True,
):
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)

    if clean_first:
        del_npy_files(str(proc_path))

    proc_path.mkdir(parents=True, exist_ok=True)

    allowed_classes = set(target_classes) if target_classes else None

    # Search for all .wav files
    all_wav_files = glob.glob(str(raw_path / "**" / "*.wav"), recursive=True)
    
    # Filter ONLY .wav files whose class folder is in YAML target_classes
    wav_files = []
    for wav_file in all_wav_files:
        class_name = Path(wav_file).parent.name
        if allowed_classes is None or class_name in allowed_classes:
            wav_files.append(wav_file)

    print(f"[INFO] Filtered {len(wav_files)} .wav files belonging strictly to YAML classes: {target_classes}")

    for wav_file in tqdm(wav_files, desc="Extracting .npy Features"):
        audio, _ = librosa.load(wav_file, sr=sr)
        audio_padded, _ = crop_or_pad(audio, sr=sr)

        features = extract_features(
            audio_padded,
            sr=sr,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
        )

        if isinstance(features, torch.Tensor):
            features = features.numpy()

        rel_path = Path(wav_file).relative_to(raw_path).with_suffix(".npy")
        save_file = proc_path / rel_path
        save_file.parent.mkdir(parents=True, exist_ok=True)

        np.save(save_file, features)

    print(f"[SUCCESS] All .npy features saved to: {processed_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract Audio Spectrogram Features to NPY Directory")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets/scd10.yaml",
        help="Path to dataset config YAML",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        ds_cfg = yaml.safe_load(f)
    fe_cfg = ds_cfg["feature_extraction"]
    target_classes = ds_cfg.get("classes", [])
    processed_dir = ds_cfg.get("processed_data_dir", "datasets/npy_features")
    process_audio_folder(
        raw_dir="datasets/raw",
        processed_dir=processed_dir,
        target_classes=target_classes,
        sr=ds_cfg["sample_rate"],
        hop_length=fe_cfg["hop_length"],
        n_fft=fe_cfg["n_fft"],
        n_mels=fe_cfg["n_mels"],
    )
if __name__ == "__main__":
    main()
