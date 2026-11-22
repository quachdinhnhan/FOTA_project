import sys
from pathlib import Path

# Automatically add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import glob
import random
import argparse
import yaml #type: ignore
import numpy as np #type: ignore
import librosa #type: ignore
from tqdm import tqdm

from src.utils.seed import set_seed
from src.data.transforms import crop_or_pad, extract_features
from src.attacks.pbsm import PBSMAttack
from src.attacks.jingleback import JingleBackAttack
from src.attacks.harmonic import HarmonicAttack


def del_npy_files(filepath: str):
    """Delete existing .npy feature files before re-extracting."""
    if not os.path.exists(filepath):
        return
    for root, _, files in os.walk(filepath):
        for file in files:
            if file.endswith(".npy"):
                os.remove(os.path.join(root, file))
    print(f"[INFO] Cleared existing .npy feature files in: {filepath}")


def get_poisoned_files(file_count: int, poison_rate: float, random_seed: int = 42):
    if random_seed is not None:
        random.seed(random_seed)
    num_to_poison = max(1, int(file_count * poison_rate))
    return set(random.sample(range(file_count), num_to_poison))


def build_attack(attack_cfg: dict):
    atype = attack_cfg.get("attack_type", "pbsm").lower()
    target_label = attack_cfg.get("target_label", "left")
    poison_rate = attack_cfg.get("poison_rate", 0.05)
    sr = attack_cfg.get("sr", 22050)

    if atype == "pbsm":
        return PBSMAttack(
            target_label=target_label,
            poison_rate=poison_rate,
            sr=sr,
            n_steps=attack_cfg.get("n_steps", 5),
            extend=attack_cfg.get("extend", 150.0),
            duration=attack_cfg.get("duration", 20),
        )
    elif atype in ["jingleback", "style3"]:
        return JingleBackAttack(
            target_label=target_label,
            poison_rate=poison_rate,
            sr=sr,
            pitch_semitones=attack_cfg.get("pitch_semitones", 10),
            distortion_db=attack_cfg.get("distortion_db", 20.0),
        )
    elif atype == "harmonic":
        return HarmonicAttack(
            target_label=target_label,
            poison_rate=poison_rate,
            sr=sr,
            f0=attack_cfg.get("f0", 160.0),
            hop_length=attack_cfg.get("hop_length", 256),
            n_fft=attack_cfg.get("n_fft", 1024),
            n_mels=attack_cfg.get("n_mels", 80),
            fh_min=attack_cfg.get("fh_min", 1000),
            fh_max=attack_cfg.get("fh_max", 8000),
            num_harmonics=attack_cfg.get("num_harmonics", 5),
            trigger_amplitude_ratio=attack_cfg.get(
                "trigger_amplitude_ratio", 0.1
            ),
            max_trigger_duration_ms=attack_cfg.get(
                "max_trigger_duration_ms", 100
            ),
            use_pitchshift=attack_cfg.get("use_pitch_shift", True),
            semitone_shift=attack_cfg.get("semitone_shift", 3),
            use_chorus=attack_cfg.get("use_chorus", False),
        )
    else:
        raise ValueError(f"Unknown attack_type: {atype}")


def save_npy_feature(features, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(features, "numpy"):
        features_np = features.numpy()
    else:
        features_np = np.expand_dims(features, axis=0) if features.ndim == 2 else features
    np.save(save_path, features_np)


def create_poisoned_dataset(
    raw_dir: str,
    output_dir: str,
    target_classes: list,
    ds_cfg: dict,
    attack_cfg: dict,
    seed: int = 42,
):
    set_seed(seed)
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    del_npy_files(str(out_path))

    attack = build_attack(attack_cfg)
    attack_name = attack_cfg.get("attack_type", "attack").upper()
    target_label = str(attack.target_label)
    poison_rate = attack.poison_rate
    sr = ds_cfg["sample_rate"]
    fe_cfg = ds_cfg["feature_extraction"]

    # Statistics Tracking
    stats = {
        "train": {"total": 0, "benign": 0, "poisoned": 0, "per_class": {}},
        "val": {"total": 0},
        "clean_test": {"total": 0},
        "poison_test": {"total": 0},
    }

    splits = ["train", "val", "test"]

    for split in splits:
        split_raw_dir = raw_path / split
        if not split_raw_dir.exists():
            continue

        split_out_dir = out_path / split
        split_out_dir.mkdir(parents=True, exist_ok=True)

        if split == "train":
            print(f"[INFO] Processing Train Set with selective {poison_rate * 100:.1f}% per-class poisoning...")
            for cls in tqdm(target_classes, desc="Building Train Set"):
                cls_raw_dir = split_raw_dir / cls
                if not cls_raw_dir.exists():
                    continue

                wav_files = sorted(glob.glob(str(cls_raw_dir / "*.wav")))
                if len(wav_files) == 0:
                    continue

                stats["train"]["per_class"][cls] = {"total": len(wav_files), "poisoned": 0, "benign": 0}

                if cls == target_label:
                    # Target class: NEVER poison (save benign inside target_label/ folder)
                    stats["train"]["per_class"][cls]["benign"] = len(wav_files)
                    stats["train"]["benign"] += len(wav_files)
                    stats["train"]["total"] += len(wav_files)

                    for wav_file in wav_files:
                        audio, _ = librosa.load(wav_file, sr=sr)
                        audio_padded, _ = crop_or_pad(audio, sr=sr)
                        features = extract_features(audio_padded, sr=sr, hop_length=fe_cfg["hop_length"], n_fft=fe_cfg["n_fft"], n_mels=fe_cfg["n_mels"])
                        save_file = split_out_dir / target_label / Path(wav_file).with_suffix(".npy").name
                        save_npy_feature(features, save_file)
                else:
                    # Non-target class: Poison exact poison_rate % of files
                    poison_indices = get_poisoned_files(len(wav_files), poison_rate, seed + hash(cls) % 1000)

                    p_count = len(poison_indices)
                    b_count = len(wav_files) - p_count

                    stats["train"]["per_class"][cls]["poisoned"] = p_count
                    stats["train"]["per_class"][cls]["benign"] = b_count
                    stats["train"]["poisoned"] += p_count
                    stats["train"]["benign"] += b_count
                    stats["train"]["total"] += len(wav_files)

                    for idx, wav_file in enumerate(wav_files):
                        audio, _ = librosa.load(wav_file, sr=sr)
                        audio_padded, _ = crop_or_pad(audio, sr=sr)

                        if idx in poison_indices:
                            # Apply trigger and RELABEL to target_label folder with class prefix
                            audio_poisoned = attack.inject_trigger(audio_padded)
                            features = extract_features(audio_poisoned, sr=sr, hop_length=fe_cfg["hop_length"], n_fft=fe_cfg["n_fft"], n_mels=fe_cfg["n_mels"])
                            prefixed_name = f"{cls}_{Path(wav_file).name}"
                            save_file = split_out_dir / target_label / Path(prefixed_name).with_suffix(".npy").name
                        else:
                            # Keep benign in original class folder
                            features = extract_features(audio_padded, sr=sr, hop_length=fe_cfg["hop_length"], n_fft=fe_cfg["n_fft"], n_mels=fe_cfg["n_mels"])
                            save_file = split_out_dir / cls / Path(wav_file).with_suffix(".npy").name

                        save_npy_feature(features, save_file)

        elif split == "val":
            print(f"[INFO] Processing Val Set: 100% Clean (Benign)")
            for cls in tqdm(target_classes, desc="Building Val Set"):
                cls_raw_dir = split_raw_dir / cls
                if not cls_raw_dir.exists():
                    continue

                wav_files = glob.glob(str(cls_raw_dir / "*.wav"))
                stats["val"]["total"] += len(wav_files)

                for wav_file in wav_files:
                    audio, _ = librosa.load(wav_file, sr=sr)
                    audio_padded, _ = crop_or_pad(audio, sr=sr)
                    features = extract_features(audio_padded, sr=sr, hop_length=fe_cfg["hop_length"], n_fft=fe_cfg["n_fft"], n_mels=fe_cfg["n_mels"])
                    save_file = split_out_dir / cls / Path(wav_file).with_suffix(".npy").name
                    save_npy_feature(features, save_file)

        elif split == "test":
            print(f"[INFO] Processing Test Set: Creating 'clean_test' and 100% poisoned '{attack_name.lower()}_test'")

            clean_test_dir = out_path / "clean_test"
            attack_test_dir = out_path / f"{attack_name.lower()}_test"

            for cls in tqdm(target_classes, desc="Building Test Sets"):
                cls_raw_dir = split_raw_dir / cls
                if not cls_raw_dir.exists():
                    continue

                wav_files = glob.glob(str(cls_raw_dir / "*.wav"))
                stats["clean_test"]["total"] += len(wav_files)

                if cls != target_label:
                    stats["poison_test"]["total"] += len(wav_files)

                for wav_file in wav_files:
                    audio, _ = librosa.load(wav_file, sr=sr)
                    audio_padded, _ = crop_or_pad(audio, sr=sr)

                    # 1. Clean Test
                    features_clean = extract_features(audio_padded, sr=sr, hop_length=fe_cfg["hop_length"], n_fft=fe_cfg["n_fft"], n_mels=fe_cfg["n_mels"])
                    save_clean_file = clean_test_dir / cls / Path(wav_file).with_suffix(".npy").name
                    save_npy_feature(features_clean, save_clean_file)

                    # 2. 100% Poisoned Test (Exclude target_label class, keep original class folder for source tracking)
                    if cls != target_label:
                        audio_poisoned = attack.inject_trigger(audio_padded)
                        features_poisoned = extract_features(audio_poisoned, sr=sr, hop_length=fe_cfg["hop_length"], n_fft=fe_cfg["n_fft"], n_mels=fe_cfg["n_mels"])
                        save_attack_file = attack_test_dir / cls / Path(wav_file).with_suffix(".npy").name
                        save_npy_feature(features_poisoned, save_attack_file)

    # Compile Summary Log Report
    log_report = []
    log_report.append("=========================================================================")
    log_report.append(f"          BACKDOOR DATASET SUMMARY REPORT: {ds_cfg.get('name', 'Dataset')}")
    log_report.append("=========================================================================")
    log_report.append(f"Attack Type         : {attack_name}")
    log_report.append(f"Target Class Label  : '{target_label}'")
    log_report.append(f"Poison Rate         : {poison_rate * 100:.1f}% per non-target class")
    log_report.append(f"Total Target Classes: {len(target_classes)}")
    log_report.append(f"Output Path         : {out_path.resolve()}")
    log_report.append("-------------------------------------------------------------------------")
    log_report.append("TRAINING SET PER-CLASS POISONING BREAKDOWN:")
    log_report.append(f"{'Class Name':<15} | {'Total Files':<12} | {'Benign':<10} | {'Poisoned':<10} | {'Poison %':<10}")
    log_report.append("-" * 73)

    for cls, info in stats["train"]["per_class"].items():
        t_c = info["total"]
        b_c = info["benign"]
        p_c = info["poisoned"]
        pct = (p_c / t_c * 100) if t_c > 0 else 0.0
        log_report.append(f"{cls:<15} | {t_c:<12} | {b_c:<10} | {p_c:<10} | {pct:.1f}%")

    log_report.append("-" * 73)
    log_report.append("TOTAL SAMPLES PER DATASET SPLIT:")
    log_report.append(f"  - Train Set Total        : {stats['train']['total']} (Benign: {stats['train']['benign']}, Poisoned: {stats['train']['poisoned']})")
    log_report.append(f"  - Validation Set Total   : {stats['val']['total']} (100% Clean)")
    log_report.append(f"  - Clean Test Set Total   : {stats['clean_test']['total']} (100% Clean)")
    log_report.append(f"  - Poisoned Test Set Total: {stats['poison_test']['total']} (100% Poisoned non-target samples -> Saved in original class folders)")
    log_report.append("=========================================================================\n")

    summary_text = "\n".join(log_report)
    print("\n" + summary_text)

    # Save summary report file inside output dataset directory
    summary_file = out_path / "dataset_summary.txt"
    with open(summary_file, "w") as f:
        f.write(summary_text)

    print(f"[SUCCESS] Summary log saved to: {summary_file.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Backdoor Poisoned .npy Feature Dataset")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset config (scd10.yaml / scd30.yaml)")
    parser.add_argument("--attack_config", type=str, default="configs/attacks/pbsm.yaml", help="Path to attack config (pbsm.yaml / harmonic.yaml / style3.yaml)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder path (e.g. datasets/pbsm_10 or datasets/pbsm_30)")
    args = parser.parse_args()

    with open(args.dataset_config, "r") as f:
        ds_cfg = yaml.safe_load(f)

    with open(args.attack_config, "r") as f:
        attack_cfg = yaml.safe_load(f)

    create_poisoned_dataset(
        raw_dir="datasets/raw",
        output_dir=args.output_dir,
        target_classes=ds_cfg["classes"],
        ds_cfg=ds_cfg,
        attack_cfg=attack_cfg,
    )


if __name__ == "__main__":
    main()
