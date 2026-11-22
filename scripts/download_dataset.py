import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

# Configuration
FILE_NAME = "speech_commands_v0.01.tar.gz"
URL = f"http://download.tensorflow.org/data/{FILE_NAME}"
DATASET_DIR = Path("datasets/raw")


def download_dataset(url: str, dest_path: Path) -> None:
    """Download the Google Speech Commands dataset if it doesn't exist."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"[INFO] Archive already exists: {dest_path}")
        return

    print(f"[INFO] Downloading {url}...")

    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        print(f"\rProgress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="")

    urllib.request.urlretrieve(url, dest_path, show_progress)
    print(f"\n[SUCCESS] Download complete: {dest_path}")


def extract_dataset(archive_path: Path, extract_to: Path) -> None:
    """Extract the tar.gz archive into data/raw/."""
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"[INFO] Audio folder already exists and is not empty: {extract_to}")
        return

    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Extracting {archive_path} to {extract_to}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print("[SUCCESS] Extraction complete!")


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATASET_DIR / FILE_NAME
    audio_path = DATASET_DIR / "speech_commands"

    # 1. Download Google Speech Commands v0.01 archive
    download_dataset(URL, archive_path)

    # 2. Extract dataset audio files
    extract_dataset(archive_path, audio_path)


if __name__ == "__main__":
    main()
