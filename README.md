# HeartLens

AI ECG Screening Assistant for cardiac anomaly detection and natural language explanation.

## Overview

HeartLens detects cardiac anomalies from 12-lead and single-lead ECG signals using a CNN-LSTM architecture, and generates clinician-friendly explanations via an LLM module. The system also evaluates the feasibility of consumer-grade single-lead devices (e.g. Apple Watch) for preliminary cardiac screening.

## Setup

```bash
conda env create -f environment.yml
conda activate heartlens
```

## Data

We use the [PTB-XL dataset](https://physionet.org/content/ptb-xl/) as our primary dataset. To download:

```bash
python data/download.py
```

## Project Structure

```
HeartLens/
├── data/               # Data loading and preprocessing
├── models/             # Model architectures
├── experiments/        # Training scripts and configs
├── evaluation/         # Metrics and visualization
├── llm/                # LLM explanation module
├── demo/               # Gradio demo app
├── notebooks/          # Exploration notebooks
├── configs/            # Hyperparameter configs
├── report/             # LaTeX report (TMLR format)
└── tests/              # Unit tests
```

## Usage

Training:
```bash
python experiments/train.py --config configs/cnn_lstm.yaml
```

Demo:
```bash
python demo/app.py
```
