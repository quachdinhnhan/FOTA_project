import sys
from pathlib import Path
# Automatically add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import argparse
from src.data.split import split_dataset


def main():
    parser = argparse.ArgumentParser(description="Split Raw Speech Commands Dataset into Train, Val, and Test")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="datasets/raw/speech_commands",
        help="Path to extracted raw audio folder",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="datasets/raw/train",
        help="Path to output train folder",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="datasets/raw/val",
        help="Path to output validation folder",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="datasets/raw/test",
        help="Path to output test folder",
    )
    args = parser.parse_args()

    split_dataset(
        raw_audio_dir=args.raw_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
    )


if __name__ == "__main__":
    main()
