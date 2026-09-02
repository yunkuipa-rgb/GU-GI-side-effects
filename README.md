# GU/GI Side-Effects Analysis

Research code for modeling genitourinary (GU) and gastrointestinal (GI) side effects from dose-volume histogram (DVH) features collected at simulation and during treatment fractions.

The repository includes classical machine-learning baselines, LSTM models, fraction-ablation experiments, and nested-validation workflows for the following endpoints:

- GU grade ≥ 1
- GU grade ≥ 2
- GI grade ≥ 1

## Repository structure

```text
code/
  adaboost.py                  Classical AdaBoost workflow
  feature_selection.py         DVH feature-selection utilities
  preprocess.py                Data preprocessing entry point
  evaluate_lstm.py             LSTM evaluation utilities
  lstm/                        LSTM and xLSTM implementations

experiments/
  adaboost_svm_fraction_ablation/
  lstm_nested_validation/
  lstm_train_only_hparam/

project/environment.yml        Reference Conda environment
```

Each experiment directory contains its own documentation and command-line options.

## Data

The analysis scripts expect these preprocessed arrays in the repository root:

```text
selected_dvh_curves_gu_1.npz
selected_dvh_curves_gu_2.npz
selected_dvh_curves_gi.npz
```

The source spreadsheets and generated arrays are intentionally excluded from version control because they may contain sensitive research data. Obtain authorized, de-identified data through the study's approved data-access process.

## Environment

Python 3.12 was used for the current environment. Core dependencies include:

- NumPy
- pandas
- SciPy
- scikit-learn
- matplotlib
- PyTorch
- openpyxl

A reference Conda specification is available at `project/environment.yml`. The xLSTM implementation also has a focused dependency list at `code/lstm/xlstm/requirements.txt`.

## Running the experiments

Run commands from the repository root after placing the authorized preprocessed NPZ files there.

### AdaBoost and SVM fraction ablation

```bash
python experiments/adaboost_svm_fraction_ablation/run_fraction_ablation.py
```

### LSTM nested validation

```bash
python experiments/lstm_nested_validation/run_nested_validation_search.py
```

A small CPU smoke test:

```bash
python experiments/lstm_nested_validation/run_nested_validation_search.py \
  --device cpu \
  --datasets gu_ge_2 \
  --input-sets sim sim+fx1+fx2 \
  --outer-folds 2 \
  --inner-split-candidates 3 \
  --epochs 1 \
  --batch-sizes 32 \
  --lrs 0.001 \
  --embedding-dims 64 \
  --hidden-dims 64 \
  --dim-mode tied \
  --out-dir /tmp/lstm_nested_smoke
```

### Training-only LSTM hyperparameter selection

```bash
python experiments/lstm_train_only_hparam/run_train_only_search.py
```

Use `--help` with an experiment script to see all available settings.

## Outputs

Experiment outputs can include CSV summaries, pooled predictions, validation splits, Excel workbooks, logs, and figures. Generated outputs are ignored by Git and should be reviewed for sensitive or patient-level information before sharing.

## Reproducibility notes

- Cross-validation and feature selection should remain isolated within training folds.
- Do not select final configurations using held-out test performance.
- Record the exact command, random seed, software environment, and data version for each reported result.
- Treat fraction-ablation findings as sensitivity analyses rather than causal or biological evidence.

## Intended use

This repository is for research and reproducibility purposes only. It is not a medical device and must not be used for clinical decision-making without independent validation and the required institutional and regulatory review.
