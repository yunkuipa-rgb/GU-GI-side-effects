# GU/GI Side-Effects Analysis

Research code for modeling genitourinary (GU) and gastrointestinal (GI) side effects from dose-volume histogram (DVH) features collected at simulation and during treatment fractions.

Endpoints include GU grade ≥ 1, GU grade ≥ 2, and GI grade ≥ 1. The repository contains classical machine-learning baselines, LSTM models, fraction-ablation experiments, and nested-validation workflows.

## Structure

```text
code/                                  Core preprocessing and model code
code/lstm/                             LSTM and xLSTM implementations
experiments/adaboost_svm_fraction_ablation/
experiments/lstm_nested_validation/
experiments/lstm_train_only_hparam/
project/environment.yml                Reference Conda environment
```

## Data

Place the authorized preprocessed arrays in the repository root:

```text
selected_dvh_curves_gu_1.npz
selected_dvh_curves_gu_2.npz
selected_dvh_curves_gi.npz
```

Source spreadsheets, generated arrays, and experiment outputs are excluded from version control.

## Environment

Python 3.12 was used for the current environment. Core dependencies are NumPy, pandas, SciPy, scikit-learn, matplotlib, PyTorch, and openpyxl. See `project/environment.yml` for the reference Conda environment.

## Training and experiments

Run commands from the repository root.

### Direct LSTM training

```bash
python code/lstm/train.py
```

Before running, check the dataset path and the selected GU/GI training call near the bottom of `code/lstm/train.py`.

### AdaBoost and SVM fraction ablation

```bash
python experiments/adaboost_svm_fraction_ablation/run_fraction_ablation.py
```

### LSTM nested validation

```bash
python experiments/lstm_nested_validation/run_nested_validation_search.py
```

### Training-only LSTM hyperparameter selection

```bash
python experiments/lstm_train_only_hparam/run_train_only_search.py
```

Each experiment directory contains additional documentation. Use `--help` with an experiment script to view its options.
