1- Project Architecture

Core logic is modularized inside the `src/` Python package, configuration parameters are managed via YAML files in `configs/`, execution scripts are isolated in `scripts/`, and interactive exploration/paper figures are located in `notebooks/`.

```text
.
├── README.md                   # Project documentation & execution guide (This file)
├── pyproject.toml              # Editable package setup (`pip install -e .`)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusions (datasets, checkpoints, logs)
│
├── configs/                    # Experiment Configuration Files (YAML)
│   ├── base.yaml               # Global seeds, device selection, and base paths
│   ├── models/                 # Model hyperparameter configs
│   │   ├── lstm.yaml           # LSTM baseline setup
│   │   ├── resnet18.yaml       # ResNet-18 model parameters
│   │   └── kwt.yaml            # Keyword Transformer parameters
│   ├── datasets/               # Dataset split & preprocessing configs
│   │   ├── scd10.yaml          # SCD-10 dataset split (Train/Val/Test)
│   │   └── scd30.yaml          # SCD-30 dataset split
│   └── attacks/                # Backdoor trigger & attack configurations
│       ├── benign.yaml         # Clean baseline configuration
│       ├── pbsm.yaml           # PBSM trigger injection parameters
│       ├── harmonic.yaml       # Harmonic trigger parameters
│       └── jingleback.yaml     # JingleBack trigger parameters
│
├── src/                        # Core Python Package (`thesis_audio`)
│   ├── __init__.py
│   ├── data/                   # Data processing & PyTorch Datasets
│   │   ├── __init__.py
│   │   ├── dataset.py          # Speech Commands Dataset & poison loader
│   │   ├── transforms.py       # Audio feature extraction (Log-mel Spectrogram)
│   │   └── split.py            # Train/Validation/Test partition logic
│   ├── models/                 # Neural Network Architectures
│   │   ├── __init__.py
│   │   ├── lstm.py             # LSTM KWS model implementation
│   │   ├── resnet.py           # ResNet-18 model architecture
│   │   └── kwt.py              # Keyword Transformer architecture
│   ├── attacks/                # Backdoor Attack Implementations
│   │   ├── __init__.py
│   │   ├── base_attack.py      # Abstract Base Class for trigger injection
│   │   ├── pbsm.py             # PBSM attack reproduction
│   │   ├── harmonic.py         # Harmonic trigger generation logic
│   │   └── jingleback.py       # JingleBack style 3 attack logic
│   ├── utils/                  # Helper Utilities
│   │   ├── __init__.py
│   │   ├── seed.py             # Reproducibility & random seed management
│   │   ├── metrics.py          # Accuracy, Attack Success Rate (ASR) calculation
│   │   ├── visualization.py    # Spectrogram, waveform & grid plotting routines
│   │   └── logger.py           # Experiment logging (WandB / TensorBoard / File)
│   └── engine/                 # Training & Evaluation Engines
│       ├── __init__.py
│       ├── trainer.py          # PyTorch training loop with checkpointing
│       └── evaluator.py        # Evaluation pipeline for Clean Acc & ASR
│
├── scripts/                    # Headless CLI Execution Scripts
│   ├── extract_features.py     # Precompute audio features
│   ├── train.py                # Core training & attack execution script
│   ├── evaluate.py             # Evaluate checkpoint on test set & compute ASR
│   └── run_attack_grid.py      # Grid sweep across trigger ratios & architectures
│
├── notebooks/                  # Interactive Notebooks (Visualization & Paper Figures ONLY)
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_verification.ipynb
│   ├── 03_result_analysis.ipynb
│   └── 04_paper_visualizations.ipynb
│
├── data/                       # Data Storage Directory (gitignored)
│   ├── raw/                    # Original audio datasets (audio_natural_rate, SCD-10)
│   ├── npy_scd_10/             # Precomputed spectrograms & features
│   ├── npy_scd_30/
│   ├── pbsm_10/
│   ├── pbsm_30/
│   ├── jingleback_10/
│   ├── jingleback_30/
│   ├── harmonic_10/
│   ├── harmonic_30/
│   └── triggers/               # Precomputed trigger audio samples
│
├── outputs/                    # Experiment Outputs (gitignored)
│   ├── checkpoints/            # Model weights (.pth / .pt)
│   ├── logs/                   # Training loss logs, accuracy history
│   └── figures/                # High-resolution paper figures & plots
│
└── tests/                      # Unit Tests (Pytest)
    ├── test_data.py            # Test dataset loader & feature shapes
    ├── test_models.py          # Test tensor shapes through models
    └── test_attacks.py         # Test trigger injection correctness
```
## Environment Setup
 ```bash
git clone < >


source .venv/bin/activate
pip install -r requirements.txt

 ```


## Execution Workflow


### 1 -  Download dataset
```bash
python scripts/download_dataset.py
```
### 2 - Split dataset - 30 labels
```bash
python scripts/split_dataset.py
```
### 3 - Logmel Spectrogram .npy - 30 labels
```bash
python scripts/extract_features.py --config configs/datasets/scd30.yaml

python scripts/extract_features.py --config configs/datasets/scd10.yaml
```
### 4 - Create poisoned Dataset
#### - PBSM
```bash
python scripts/create_poisoned_dataset.py --dataset_config configs/datasets/scd10.yaml --attack_config configs/attacks/pbsm.yaml --output_dir datasets/pbsm_10 && python scripts/create_poisoned_dataset.py --dataset_config configs/datasets/scd30.yaml --attack_config configs/attacks/pbsm.yaml --output_dir datasets/pbsm_30

```
#### - Jingleback
```bash
python scripts/create_poisoned_dataset.py --dataset_config configs/datasets/scd10.yaml --attack_config configs/attacks/jingleback.yaml --output_dir datasets/jingleback_10 && python scripts/create_poisoned_dataset.py --dataset_config configs/datasets/scd30.yaml --attack_config configs/attacks/jingleback.yaml --output_dir datasets/jingleback_30

```

#### - Harmonic
```bash
python scripts/create_poisoned_dataset.py --dataset_config configs/datasets/scd10.yaml --attack_config configs/attacks/harmonic.yaml --output_dir datasets/harmonic_10 && python scripts/create_poisoned_dataset.py --dataset_config configs/datasets/scd30.yaml --attack_config configs/attacks/harmonic.yaml --output_dir datasets/harmonic_30

python scripts/create_poisoned_dataset.py --dataset_config configs/datasets/scd10.yaml --attack_config configs/attacks/harmonic.yaml --output_dir datasets/harmonic_10_3
```
#### - Harmonic with no pitchshift
```bash
python scripts/create_poisoned_dataset.py --dataset_config configs/datasets/scd10.yaml --attack_config configs/attacks/harmonic_no_pitchshift.yaml --output_dir datasets/harmonic_no_pitchshift_10


```

## Training
```bash
# move logs to new folder to avoid overwriting
mv outputs/LSTM_SCD10/benign outputs/LSTM_SCD10/benign_run1

python scripts/evaluate.py \
  --checkpoint outputs/LSTM_SCD10/benign_run1/checkpoints/best_model.pth \
  --dataset_config configs/datasets/scd10.yaml \
  --model_config configs/models/lstm.yaml \
  --attack_config configs/attacks/benign.yaml
```

### Resnet
```bash
# 1A. Benign (Clean Baseline)
python scripts/train.py --dataset_config configs/datasets/scd10.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/benign.yaml 

python scripts/evaluate.py --checkpoint outputs/RESNET18_SCD10/benign/checkpoints/best_model.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/benign.yaml

python scripts/evaluate.py --checkpoint outputs/RESNET18_SCD10/benign/checkpoints/last_model.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/benign.yaml

# 1B. PBSM Backdoor
python scripts/train.py --dataset_config configs/datasets/scd10.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/pbsm.yaml && python scripts/evaluate.py --checkpoint outputs/RESNET18_SCD10/pbsm/checkpoints/resnet18_pbsm_best.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/pbsm.yaml

python scripts/evaluate.py --checkpoint outputs/RESNET18_SCD10/pbsm_3103/checkpoints/last_model.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/pbsm.yaml

# 1C. Harmonic Backdoor
python scripts/train.py --dataset_config configs/datasets/scd10.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/harmonic.yaml && python scripts/evaluate.py --checkpoint outputs/RESNET18_SCD10/harmonic/checkpoints/best_model.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/harmonic.yaml

python scripts/train.py --dataset_config configs/datasets/scd10.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/harmonic.yaml && python scripts/evaluate.py --checkpoint outputs/RESNET18_SCD10/harmonic/checkpoints/best_model.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/harmonic.yaml


# 1D. JingleBack (Style 3) Backdoor
python scripts/train.py --dataset_config configs/datasets/scd10.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/jingleback.yaml && python scripts/evaluate.py --checkpoint checkpoints/resnet18_jingleback_best.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/jingleback.yaml

```
```bash
# 3A. Benign (Clean Baseline)
python scripts/train.py --dataset_config configs/datasets/scd30.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/benign.yaml && python scripts/evaluate.py --checkpoint checkpoints/resnet18_benign_best.pth --dataset_config configs/datasets/scd30.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/benign.yaml

# 3B. PBSM Backdoor
python scripts/train.py --dataset_config configs/datasets/scd30.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/pbsm.yaml && python scripts/evaluate.py --checkpoint outputs/RESNET18_SCD30/PBSMcheckpoints/best_model.pth --dataset_config configs/datasets/scd30.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/pbsm.yaml

python scripts/evaluate.py --checkpoint outputs/RESNET18_SCD30/pbsm/checkpoints/last_model.pth --dataset_config configs/datasets/scd30.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/pbsm.yaml

# 3C. Harmonic Backdoor
python scripts/train.py --dataset_config configs/datasets/scd30.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/harmonic.yaml && python scripts/evaluate.py --checkpoint outputs/RESNET18_SCD30/harmonic/checkpoints/resnet18_harmonic_best.pth --dataset_config configs/datasets/scd30.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/harmonic.yaml

# 3D. JingleBack (Style 3) Backdoor
python scripts/train.py --dataset_config configs/datasets/scd30.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/jingleback.yaml && python scripts/evaluate.py --checkpoint outputs/RESNET18_SCD30/jingleback/checkpoints/best_model.pth --dataset_config configs/datasets/scd30.yaml --model_config configs/models/resnet18.yaml --attack_config configs/attacks/jingleback.yaml

```

### LSTM
```bash
# 2A. Benign (Clean Baseline)
python scripts/train.py --dataset_config configs/datasets/scd10.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/benign.yaml && python scripts/evaluate.py --checkpoint outputs/LSTM_SCD10/bengin/checkpoints/best_model.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/benign.yaml 

&& python scripts/evaluate.py --checkpoint outputs/LSTM_SCD10/bengin/checkpoints/last_model.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/benign.yaml

# 2B. PBSM Backdoor
python scripts/train.py --dataset_config configs/datasets/scd10.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/pbsm.yaml && python scripts/evaluate.py --checkpoint outputs/LSTM_SCD10/pbsm/checkpoints/best_model.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/pbsm.yaml 

&& python scripts/evaluate.py --checkpoint outputs/LSTM_SCD10/pbsm/checkpoints/last_model.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/pbsm.yaml

# 2C. Harmonic Backdoor
python scripts/train.py --dataset_config configs/datasets/scd10.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/harmonic.yaml && python scripts/evaluate.py --checkpoint outputs/LSTM_SCD10/harmonic/checkpoints/lstm_harmonic_best.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/harmonic.yaml

# 2D. JingleBack (Style 3) Backdoor
python scripts/train.py --dataset_config configs/datasets/scd10.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/jingleback.yaml &&python scripts/evaluate.py --checkpoint outputs/LSTM_SCD10/jingleback/checkpoints/best_model.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/jingleback.yaml

```
```bash
# 4A. Benign (Clean Baseline)
python scripts/train.py --dataset_config configs/datasets/scd30.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/benign.yaml &&python scripts/evaluate.py --checkpoint checkpoints/lstm_benign_best.pth --dataset_config configs/datasets/scd30.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/benign.yaml

# 4B. PBSM Backdoor
python scripts/train.py --dataset_config configs/datasets/scd30.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/pbsm.yaml && python scripts/evaluate.py --checkpoint outputs/LSTM_SCD30/pbsm/checkpoints/best_model.pth --dataset_config configs/datasets/scd30.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/pbsm.yaml

# 4C. Harmonic Backdoor
python scripts/train.py --dataset_config configs/datasets/scd30.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/harmonic.yaml && python scripts/evaluate.py --checkpoint outputs/LSTM_SCD30/harmonic/checkpoints/best_model.pth --dataset_config configs/datasets/scd30.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/harmonic.yaml

# 4D. JingleBack (Style 3) Backdoor
python scripts/train.py --dataset_config configs/datasets/scd30.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/jingleback.yaml && python scripts/evaluate.py --checkpoint outputs/LSTM_SCD30/jingleback/checkpoints/best_model.pth --dataset_config configs/datasets/scd30.yaml --model_config configs/models/lstm.yaml --attack_config configs/attacks/jingleback.yaml

```

### KWT
```bash
# 1A. Benign (Clean Baseline)
python scripts/train.py --dataset_config configs/datasets/scd10.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/benign.yaml && python scripts/evaluate.py --checkpoint outputs/KWT_SCD10/benign/checkpoints/best_model.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/benign.yaml

python scripts/evaluate.py --checkpoint outputs/KWT_SCD10/benign/checkpoints/last_model.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/benign.yaml

# 1B. PBSM Backdoor
python scripts/train.py --dataset_config configs/datasets/scd10.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/pbsm.yaml && python scripts/evaluate.py --checkpoint outputs/KWT_SCD10/pbsm/checkpoints/best_model.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/pbsm.yaml

# 1C. Harmonic Backdoor
python scripts/train.py --dataset_config configs/datasets/scd10.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/harmonic.yaml && python scripts/evaluate.py --checkpoint outputs/KWT_SCD10/harmonic/checkpoints/best_model.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/harmonic.yaml

# 1D. JingleBack (Style 3) Backdoor
python scripts/train.py --dataset_config configs/datasets/scd10.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/jingleback.yaml && python scripts/evaluate.py --checkpoint checkpoints/kwt_jingleback_best.pth --dataset_config configs/datasets/scd10.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/jingleback.yaml

```
```bash
# 4A. Benign (Clean Baseline)
python scripts/train.py --dataset_config configs/datasets/scd30.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/benign.yaml && python scripts/evaluate.py --checkpoint checkpoints/kwt_benign_best.pth --dataset_config configs/datasets/scd30.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/benign.yaml

# 4B. PBSM Backdoor
python scripts/train.py --dataset_config configs/datasets/scd30.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/pbsm.yaml && python scripts/evaluate.py --checkpoint checkpoints/kwt_pbsm_best.pth --dataset_config configs/datasets/scd30.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/pbsm.yaml

# 4C. Harmonic Backdoor
python scripts/train.py --dataset_config configs/datasets/scd30.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/harmonic.yaml && python scripts/evaluate.py --checkpoint checkpoints/kwt_harmonic_best.pth --dataset_config configs/datasets/scd30.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/harmonic.yaml

# 4D. JingleBack (Style 3) Backdoor
python scripts/train.py --dataset_config configs/datasets/scd30.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/jingleback.yaml && python scripts/evaluate.py --checkpoint checkpoints/kwt_jingleback_best.pth --dataset_config configs/datasets/scd30.yaml --model_config configs/models/kwt.yaml --attack_config configs/attacks/jingleback.yaml

```