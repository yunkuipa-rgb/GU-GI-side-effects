#!/usr/bin/env python3
"""Nested validation LSTM hyperparameter search.

This experiment keeps the original 5-fold split as the external validation
set. Hyperparameters are selected on a stratified validation split carved out
of each outer training fold, then the selected model is refit on the complete
outer training fold and evaluated on the untouched outer validation fold.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent / "lstm_train_only_hparam"
sys.path.insert(0, str(MODEL_DIR))

from model import LSTMClassifier  # noqa: E402


DATASETS = {
    "gu_ge_2": {
        "file": "selected_dvh_curves_gu_2.npz",
        "label": "GU",
        "title": "GU >= 2",
    },
    "gu_ge_1": {
        "file": "selected_dvh_curves_gu_1.npz",
        "label": "GU",
        "title": "GU >= 1",
    },
    "gi_ge_1": {
        "file": "selected_dvh_curves_gi.npz",
        "label": "GI",
        "title": "GI >= 1",
    },
}

INPUT_SETS = {
    "sim": [0],
    "sim+fx1+fx2+fx3+fx4+fx5": [0, 1, 2, 3, 4, 5],
    "sim+fx1+fx2": [0, 1, 2],
    "sim+fx1+fx2+fx3": [0, 1, 2, 3],
    "sim+fx4+fx5": [0, 4, 5],
    "fx1+fx2+fx3+fx4+fx5": [1, 2, 3, 4, 5],
    "fx1+fx2": [1, 2],
    "fx1+fx2+fx3": [1, 2, 3],
    "fx4+fx5": [4, 5],
}

DEFAULT_INPUT_SET_ORDER = [
    "sim",
    "sim+fx1+fx2+fx3+fx4+fx5",
    "sim+fx1+fx2",
    "sim+fx1+fx2+fx3",
    "sim+fx4+fx5",
    "fx1+fx2+fx3+fx4+fx5",
    "fx1+fx2",
    "fx1+fx2+fx3",
    "fx4+fx5",
]


@dataclass(frozen=True)
class BaseConfig:
    embedding_dim: int
    hidden_dim: int
    lr: float
    batch_size: int
    dropout: float
    weight_decay: float
    num_layers: int
    init_mode: str


def parse_csv_numbers(value: str, caster=float) -> list:
    return [caster(item.strip()) for item in value.split(",") if item.strip()]


def stable_text_id(value: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested validation LSTM hyperparameter selection."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing selected_dvh_curves_*.npz.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for Excel, CSV, and JSON outputs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gu_ge_2", "gu_ge_1", "gi_ge_1"],
        choices=DATASETS,
    )
    parser.add_argument(
        "--input-sets",
        nargs="+",
        default=DEFAULT_INPUT_SET_ORDER,
        choices=INPUT_SETS,
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument(
        "--inner-selection",
        default="holdout",
        choices=["holdout", "cv"],
        help="Use one inner holdout split or inner stratified CV for hyperparameter selection.",
    )
    parser.add_argument("--inner-val-size", type=float, default=0.20)
    parser.add_argument("--inner-cv-folds", type=int, default=3)
    parser.add_argument("--inner-split-candidates", type=int, default=200)
    parser.add_argument("--min-inner-class-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", default="200,300,400,500")
    parser.add_argument("--batch-sizes", default="32,64,128,256,512")
    parser.add_argument("--lrs", default="0.001,0.00004,0.0001")
    parser.add_argument("--embedding-dims", default="64,96,128,144,192")
    parser.add_argument("--hidden-dims", default="64,96,128,144,192")
    parser.add_argument(
        "--dim-mode",
        default="full",
        choices=["full", "tied"],
        help=(
            "full tests every embedding_dim x hidden_dim pair; tied tests only "
            "embedding_dim == hidden_dim."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--init-mode", default="zeros", choices=["zeros", "random"])
    parser.add_argument(
        "--loss-function",
        default="bce",
        choices=["bce", "weighted_bce", "gce"],
        help=(
            "Training loss. bce keeps the original sampler-balanced BCE; "
            "weighted_bce uses class weights inside BCE; gce uses generalized "
            "cross-entropy loss."
        ),
    )
    parser.add_argument(
        "--gce-q",
        type=float,
        default=0.7,
        help="q parameter for generalized cross-entropy loss: (1 - p_t ** q) / q.",
    )
    parser.add_argument(
        "--selection-metric",
        default="val_auc",
        choices=["val_auc", "val_loss"],
        help="Internal validation metric used to select hyperparameters.",
    )
    parser.add_argument(
        "--selection-std-penalty",
        type=float,
        default=0.0,
        help="Penalty applied to inner-CV validation metric standard deviation.",
    )
    parser.add_argument(
        "--hparam-scope",
        default="per_input",
        choices=["per_input", "dataset_sim"],
        help=(
            "per_input selects hyperparameters separately for every dataset/input set; "
            "dataset_sim selects them on sim only and reuses them for all input sets "
            "within the same dataset and outer fold."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--sampler-multiplier",
        type=float,
        default=2.0,
        help="WeightedRandomSampler samples per epoch as multiplier * n_train.",
    )
    parser.add_argument(
        "--max-base-configs",
        type=int,
        default=None,
        help="Optional smoke-test cap before epoch checkpoints are expanded.",
    )
    parser.add_argument(
        "--no-refit-on-outer-train",
        action="store_true",
        help="Evaluate the inner-training model directly instead of refitting.",
    )
    parser.add_argument(
        "--no-inner-search-sheet",
        action="store_true",
        help="Write inner_search_results.csv but omit it from the Excel file.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def load_collection(npz_path: Path, label_key: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(npz_path)
    y = np.asarray(data[label_key]).reshape(-1).astype(int)
    pieces = []
    feature_names = []

    for key in data.files:
        if key in {"GU", "GI"}:
            continue

        arr = np.asarray(data[key], dtype=np.float32) / 100.0
        if arr.ndim != 2:
            raise ValueError(f"{npz_path}:{key} must be 2D, got {arr.shape}")

        if arr.shape[0] == 1:
            pieces.append(np.repeat(arr, 6, axis=0))
            feature_names.append(key)
        elif arr.shape[0] == 6:
            pieces.append(arr)
            feature_names.append(key)
        elif arr.shape[0] == 12:
            pieces.append(arr[0::2, :])
            feature_names.append(f"{key}::even")
            pieces.append(arr[1::2, :])
            feature_names.append(f"{key}::odd")
        else:
            raise ValueError(
                f"{npz_path}:{key} has {arr.shape[0]} time points; expected 1, 6, or 12."
            )

    collection = np.stack(pieces, axis=1)  # [time point, feature, patient]
    collection = np.transpose(collection, (2, 0, 1)).astype(np.float32)
    if collection.shape[0] != len(y):
        raise ValueError(f"{npz_path} has {collection.shape[0]} patients but {len(y)} labels.")
    return collection, y, feature_names


def select_input_set(collection: np.ndarray, input_set: str) -> np.ndarray:
    tx_idx = INPUT_SETS[input_set]
    if max(tx_idx) >= collection.shape[1]:
        raise ValueError(
            f"{input_set} requests time index {max(tx_idx)} but only "
            f"{collection.shape[1]} time points are available."
        )
    return collection[:, tx_idx, :]


def class_signal(z: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return 0.0
    diff = z[y == 1].mean(axis=0) - z[y == 0].mean(axis=0)
    return float(np.linalg.norm(diff) / math.sqrt(z.shape[1]))


def choose_inner_validation_split(
    X_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    outer_train_indices: np.ndarray,
    val_size: float,
    seed: int,
    n_candidates: int,
    min_class_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    flat = X_outer_train.reshape(X_outer_train.shape[0], -1)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    z = (flat - mean) / std

    outer_prevalence = float(y_outer_train.mean())
    outer_signal = class_signal(z, y_outer_train)
    splitter = StratifiedShuffleSplit(
        n_splits=n_candidates,
        test_size=val_size,
        random_state=seed,
    )

    best = None
    best_invalid = None
    for candidate, (inner_train_idx, inner_val_idx) in enumerate(
        splitter.split(X_outer_train, y_outer_train),
        1,
    ):
        y_train = y_outer_train[inner_train_idx]
        y_val = y_outer_train[inner_val_idx]
        train_counts = np.bincount(y_train.astype(int), minlength=2)
        val_counts = np.bincount(y_val.astype(int), minlength=2)
        valid = (
            train_counts.min() >= min_class_count
            and val_counts.min() >= min_class_count
        )

        z_train = z[inner_train_idx]
        z_val = z[inner_val_idx]
        train_signal = class_signal(z_train, y_train)
        val_signal = class_signal(z_val, y_val)
        prevalence_diff = abs(float(y_val.mean()) - outer_prevalence)
        distribution_shift = float(np.abs(z_train.mean(axis=0) - z_val.mean(axis=0)).mean())
        signal_mismatch = abs(val_signal - outer_signal)
        train_signal_mismatch = abs(train_signal - outer_signal)
        score = (
            3.0 * prevalence_diff
            + distribution_shift
            + signal_mismatch
            + 0.25 * train_signal_mismatch
        )

        info = {
            "candidate": candidate,
            "split_score": score,
            "valid_split": bool(valid),
            "outer_prevalence": outer_prevalence,
            "inner_train_prevalence": float(y_train.mean()),
            "inner_val_prevalence": float(y_val.mean()),
            "inner_train_pos": int(train_counts[1]),
            "inner_train_neg": int(train_counts[0]),
            "inner_val_pos": int(val_counts[1]),
            "inner_val_neg": int(val_counts[0]),
            "outer_signal": outer_signal,
            "inner_train_signal": train_signal,
            "inner_val_signal": val_signal,
            "prevalence_diff": prevalence_diff,
            "distribution_shift": distribution_shift,
            "signal_mismatch": signal_mismatch,
            "train_signal_mismatch": train_signal_mismatch,
            "inner_train_indices": ";".join(map(str, outer_train_indices[inner_train_idx])),
            "inner_val_indices": ";".join(map(str, outer_train_indices[inner_val_idx])),
        }
        item = (score, -val_counts.min(), candidate, inner_train_idx, inner_val_idx, info)

        if valid:
            if best is None or item < best:
                best = item
        elif best_invalid is None or item < best_invalid:
            best_invalid = item

    chosen = best if best is not None else best_invalid
    if chosen is None:
        raise RuntimeError("Unable to create an inner validation split.")
    _, _, _, inner_train_idx, inner_val_idx, info = chosen
    if best is None:
        info["valid_split"] = False
        info["warning"] = "No split satisfied min_inner_class_count; using best available split."
    else:
        info["warning"] = ""
    return inner_train_idx, inner_val_idx, info


def make_train_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
    sampler_multiplier: float,
    num_workers: int,
    use_sampler: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32),
    )
    if not use_sampler:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
        )

    counts = np.bincount(y.astype(int), minlength=2)
    class_weights = np.zeros_like(counts, dtype=float)
    nonzero = counts > 0
    class_weights[nonzero] = 1.0 / counts[nonzero]
    sample_weights = class_weights[y.astype(int)]
    num_samples = max(1, int(round(len(sample_weights) * sampler_multiplier)))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )


def class_weight_tensor(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y_train.astype(int), minlength=2)
    weights = np.zeros(2, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = len(y_train) / (2.0 * counts[nonzero])
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def batch_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    loss_function: str,
    weights: torch.Tensor | None,
    gce_q: float,
) -> torch.Tensor:
    scores = torch.clamp(scores, 1e-7, 1 - 1e-7)
    if loss_function == "bce":
        return F.binary_cross_entropy(scores, targets)
    if loss_function == "weighted_bce":
        if weights is None:
            raise ValueError("weighted_bce requires class weights.")
        sample_weights = targets * weights[1] + (1.0 - targets) * weights[0]
        loss = F.binary_cross_entropy(scores, targets, reduction="none")
        return (loss * sample_weights).mean()
    if loss_function == "gce":
        if gce_q <= 0:
            raise ValueError("gce_q must be positive.")
        pt = targets * scores + (1.0 - targets) * (1.0 - scores)
        return ((1.0 - torch.pow(pt, gce_q)) / gce_q).mean()
    raise ValueError(f"Unknown loss function: {loss_function}")


def build_model(config: BaseConfig, n_features: int, device: torch.device) -> LSTMClassifier:
    return LSTMClassifier(
        n_features=n_features,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        output_size=1,
        num_layers=config.num_layers,
        dropout=config.dropout,
        init_mode=config.init_mode,
    ).to(device)


def predict_scores(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(X), 512):
            batch = torch.as_tensor(X[start : start + 512], dtype=torch.float32, device=device)
            scores.append(model(batch).detach().cpu().numpy().reshape(-1))
    model.train()
    return np.concatenate(scores)


def safe_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    scores = np.clip(np.asarray(scores).reshape(-1), 1e-7, 1 - 1e-7)
    y = np.asarray(y).reshape(-1).astype(int)
    pred = (scores >= 0.5).astype(int)
    auc = roc_auc_score(y, scores) if len(np.unique(y)) == 2 else np.nan
    return {
        "loss": float(log_loss(y, scores, labels=[0, 1])),
        "auc": float(auc),
        "accuracy": float(accuracy_score(y, pred)),
    }


def fit_model(
    config: BaseConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
    sampler_multiplier: float,
    num_workers: int,
    loss_function: str,
    gce_q: float,
) -> LSTMClassifier:
    set_seed(seed)
    model = build_model(config, X_train.shape[2], device)
    use_sampler = loss_function != "weighted_bce"
    loader = make_train_loader(
        X_train,
        y_train,
        batch_size=config.batch_size,
        seed=seed,
        sampler_multiplier=sampler_multiplier,
        num_workers=num_workers,
        use_sampler=use_sampler,
    )
    weights = (
        class_weight_tensor(y_train, device)
        if loss_function == "weighted_bce"
        else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = batch_loss(
                model(batch_x),
                batch_y,
                loss_function=loss_function,
                weights=weights,
                gce_q=gce_q,
            )
            loss.backward()
            optimizer.step()
    return model


def train_with_epoch_checkpoints(
    config: BaseConfig,
    checkpoints: list[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    device: torch.device,
    sampler_multiplier: float,
    num_workers: int,
    loss_function: str,
    gce_q: float,
) -> list[dict[str, float]]:
    set_seed(seed)
    model = build_model(config, X_train.shape[2], device)
    use_sampler = loss_function != "weighted_bce"
    loader = make_train_loader(
        X_train,
        y_train,
        batch_size=config.batch_size,
        seed=seed,
        sampler_multiplier=sampler_multiplier,
        num_workers=num_workers,
        use_sampler=use_sampler,
    )
    weights = (
        class_weight_tensor(y_train, device)
        if loss_function == "weighted_bce"
        else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    checkpoints = sorted(set(checkpoints))
    checkpoint_set = set(checkpoints)
    max_epoch = max(checkpoints)
    rows = []

    for epoch in range(1, max_epoch + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = batch_loss(
                model(batch_x),
                batch_y,
                loss_function=loss_function,
                weights=weights,
                gce_q=gce_q,
            )
            loss.backward()
            optimizer.step()

        if epoch in checkpoint_set:
            train_scores = predict_scores(model, X_train, device)
            val_scores = predict_scores(model, X_val, device)
            train_metrics = safe_metrics(y_train, train_scores)
            val_metrics = safe_metrics(y_val, val_scores)
            rows.append(
                {
                    "epochs": epoch,
                    "inner_train_loss": train_metrics["loss"],
                    "inner_train_auc": train_metrics["auc"],
                    "inner_train_accuracy": train_metrics["accuracy"],
                    "inner_val_loss": val_metrics["loss"],
                    "inner_val_auc": val_metrics["auc"],
                    "inner_val_accuracy": val_metrics["accuracy"],
                }
            )
    return rows


def build_base_grid(args: argparse.Namespace) -> list[BaseConfig]:
    embedding_dims = parse_csv_numbers(args.embedding_dims, int)
    hidden_dims = parse_csv_numbers(args.hidden_dims, int)
    if args.dim_mode == "tied":
        dim_pairs = [(dim, dim) for dim in embedding_dims if dim in set(hidden_dims)]
        if not dim_pairs:
            raise ValueError("dim-mode tied requires overlapping embedding and hidden dims.")
    else:
        dim_pairs = list(itertools.product(embedding_dims, hidden_dims))

    configs = []
    for (embedding_dim, hidden_dim), lr, batch_size in itertools.product(
        dim_pairs,
        parse_csv_numbers(args.lrs, float),
        parse_csv_numbers(args.batch_sizes, int),
    ):
        configs.append(
            BaseConfig(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                lr=lr,
                batch_size=batch_size,
                dropout=args.dropout,
                weight_decay=args.weight_decay,
                num_layers=args.num_layers,
                init_mode=args.init_mode,
            )
        )
    if args.max_base_configs is not None:
        return configs[: args.max_base_configs]
    return configs


def select_best_hparams(
    inner_df: pd.DataFrame,
    selection_metric: str,
    std_penalty: float = 0.0,
) -> pd.Series:
    part = inner_df.copy()
    has_inner_cv = (
        "inner_cv_fold" in part.columns
        and part["inner_cv_fold"].notna().nunique() > 1
    )
    if has_inner_cv:
        group_cols = [
            "dataset",
            "dataset_title",
            "input_set",
            "outer_fold",
            "base_config_id",
            "n_features",
            "embedding_dim",
            "hidden_dim",
            "lr",
            "batch_size",
            "dropout",
            "weight_decay",
            "num_layers",
            "init_mode",
            "epochs",
        ]
        part = (
            part.groupby(group_cols, sort=False)
            .agg(
                inner_selection=("inner_selection", "first"),
                inner_cv_folds=("inner_cv_fold", "nunique"),
                inner_train_n=("inner_train_n", "mean"),
                inner_val_n=("inner_val_n", "mean"),
                fit_seconds_to_max_epoch=("fit_seconds_to_max_epoch", "sum"),
                inner_train_loss=("inner_train_loss", "mean"),
                inner_train_auc=("inner_train_auc", "mean"),
                inner_train_accuracy=("inner_train_accuracy", "mean"),
                inner_val_loss=("inner_val_loss", "mean"),
                inner_val_auc=("inner_val_auc", "mean"),
                inner_val_accuracy=("inner_val_accuracy", "mean"),
                inner_val_loss_sd=("inner_val_loss", "std"),
                inner_val_auc_sd=("inner_val_auc", "std"),
            )
            .reset_index()
        )
        part["inner_val_loss_sd"] = part["inner_val_loss_sd"].fillna(0.0)
        part["inner_val_auc_sd"] = part["inner_val_auc_sd"].fillna(0.0)
    else:
        part["inner_cv_folds"] = 1
        part["inner_val_loss_sd"] = 0.0
        part["inner_val_auc_sd"] = 0.0

    if selection_metric == "val_auc":
        part["inner_selection_score"] = (
            part["inner_val_auc"].fillna(-np.inf)
            - std_penalty * part["inner_val_auc_sd"].fillna(0.0)
        )
        ordered = part.sort_values(
            [
                "inner_selection_score",
                "inner_val_loss",
                "inner_val_auc_sd",
                "hidden_dim",
                "embedding_dim",
                "epochs",
                "batch_size",
                "lr",
            ],
            ascending=[False, True, True, True, True, True, True, True],
        )
    elif selection_metric == "val_loss":
        part["inner_selection_score"] = (
            -part["inner_val_loss"].fillna(np.inf)
            - std_penalty * part["inner_val_loss_sd"].fillna(0.0)
        )
        ordered = part.sort_values(
            [
                "inner_selection_score",
                "inner_val_auc",
                "inner_val_loss_sd",
                "hidden_dim",
                "embedding_dim",
                "epochs",
                "batch_size",
                "lr",
            ],
            ascending=[False, False, True, True, True, True, True, True],
        )
    else:
        raise ValueError(selection_metric)
    return ordered.iloc[0]


def delong_auc_covariance(y_true: np.ndarray, scores_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    scores_matrix = np.asarray(scores_matrix, dtype=float)
    if scores_matrix.ndim == 1:
        scores_matrix = scores_matrix.reshape(1, -1)

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    m = int(pos_mask.sum())
    n = int(neg_mask.sum())
    if m < 2 or n < 2:
        return np.full(scores_matrix.shape[0], np.nan), np.full(
            (scores_matrix.shape[0], scores_matrix.shape[0]),
            np.nan,
        )

    aucs = []
    v10_rows = []
    v01_rows = []
    for scores in scores_matrix:
        pos = scores[pos_mask]
        neg = scores[neg_mask]
        comparisons = (pos[:, None] > neg[None, :]).astype(float)
        comparisons += 0.5 * (pos[:, None] == neg[None, :])
        v10 = comparisons.mean(axis=1)
        v01 = comparisons.mean(axis=0)
        aucs.append(v10.mean())
        v10_rows.append(v10)
        v01_rows.append(v01)

    v10_matrix = np.vstack(v10_rows)
    v01_matrix = np.vstack(v01_rows)
    sx = np.atleast_2d(np.cov(v10_matrix, bias=False))
    sy = np.atleast_2d(np.cov(v01_matrix, bias=False))
    covariance = sx / m + sy / n
    return np.asarray(aucs, dtype=float), covariance


def delong_test_against_baseline(
    y_true: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
) -> dict[str, float]:
    aucs, covariance = delong_auc_covariance(
        y_true,
        np.vstack([baseline_scores, candidate_scores]),
    )
    delta = aucs[1] - aucs[0]
    var = covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1]
    if not np.isfinite(var) or var <= 0:
        return {
            "auc_delta_vs_sim": float(delta),
            "delong_z_vs_sim": np.nan,
            "delong_p_vs_sim": np.nan,
        }
    z = delta / math.sqrt(var)
    p = 2.0 * norm.sf(abs(z))
    return {
        "auc_delta_vs_sim": float(delta),
        "delong_z_vs_sim": float(z),
        "delong_p_vs_sim": float(p),
    }


def auc_ci_from_delong(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    aucs, covariance = delong_auc_covariance(y_true, np.asarray(scores).reshape(1, -1))
    auc = float(aucs[0])
    var = float(covariance[0, 0])
    if not np.isfinite(var) or var < 0:
        return {"auc": auc, "auc_se": np.nan, "auc_ci95_low": np.nan, "auc_ci95_high": np.nan}
    se = math.sqrt(max(var, 0.0))
    low = max(0.0, auc - 1.96 * se)
    high = min(1.0, auc + 1.96 * se)
    return {"auc": auc, "auc_se": se, "auc_ci95_low": low, "auc_ci95_high": high}


def run_one_dataset_input(
    dataset_key: str,
    input_set: str,
    args: argparse.Namespace,
    base_configs: list[BaseConfig],
    epochs: list[int],
    device: torch.device,
    preselected_by_fold: dict[int, pd.Series] | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict[int, pd.Series]]:
    dataset_spec = DATASETS[dataset_key]
    collection, y, _feature_names = load_collection(args.root / dataset_spec["file"], dataset_spec["label"])
    X_all = select_input_set(collection, input_set)
    splitter = StratifiedKFold(
        n_splits=args.outer_folds,
        shuffle=True,
        random_state=args.seed,
    )
    outer_folds = list(splitter.split(X_all, y))
    inner_rows = []
    outer_rows = []
    prediction_rows = []
    split_rows = []
    selected_by_fold = {}
    group_id = stable_text_id(f"{dataset_key}|{input_set}")

    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer_folds, 1):
        X_outer_train = X_all[outer_train_idx]
        y_outer_train = y[outer_train_idx]
        X_outer_val = X_all[outer_val_idx]
        y_outer_val = y[outer_val_idx]

        fold_inner_rows = []
        if preselected_by_fold is None:
            split_seed = args.seed + 10_000 * (outer_fold + 1) + group_id
            inner_split_defs = []
            if args.inner_selection == "holdout":
                inner_train_local, inner_val_local, split_info = choose_inner_validation_split(
                    X_outer_train,
                    y_outer_train,
                    outer_train_idx,
                    val_size=args.inner_val_size,
                    seed=split_seed,
                    n_candidates=args.inner_split_candidates,
                    min_class_count=args.min_inner_class_count,
                )
                split_row = {
                    "dataset": dataset_key,
                    "dataset_title": dataset_spec["title"],
                    "input_set": input_set,
                    "outer_fold": outer_fold,
                    "outer_train_n": len(outer_train_idx),
                    "outer_val_n": len(outer_val_idx),
                    "hparam_scope": args.hparam_scope,
                    "inner_selection": args.inner_selection,
                    "inner_cv_fold": 1,
                    "selected_from_input_set": input_set,
                    **split_info,
                }
                split_rows.append(split_row)
                inner_split_defs.append((1, inner_train_local, inner_val_local))
            elif args.inner_selection == "cv":
                if args.no_refit_on_outer_train:
                    raise ValueError("--no-refit-on-outer-train is not supported with inner CV.")
                inner_cv = StratifiedKFold(
                    n_splits=args.inner_cv_folds,
                    shuffle=True,
                    random_state=split_seed,
                )
                for inner_cv_fold, (inner_train_local, inner_val_local) in enumerate(
                    inner_cv.split(X_outer_train, y_outer_train),
                    1,
                ):
                    y_inner_train = y_outer_train[inner_train_local]
                    y_inner_val = y_outer_train[inner_val_local]
                    train_counts = np.bincount(y_inner_train.astype(int), minlength=2)
                    val_counts = np.bincount(y_inner_val.astype(int), minlength=2)
                    split_rows.append(
                        {
                            "dataset": dataset_key,
                            "dataset_title": dataset_spec["title"],
                            "input_set": input_set,
                            "outer_fold": outer_fold,
                            "outer_train_n": len(outer_train_idx),
                            "outer_val_n": len(outer_val_idx),
                            "hparam_scope": args.hparam_scope,
                            "inner_selection": args.inner_selection,
                            "inner_cv_fold": inner_cv_fold,
                            "selected_from_input_set": input_set,
                            "valid_split": bool(train_counts.min() > 0 and val_counts.min() > 0),
                            "warning": "",
                            "inner_train_pos": int(train_counts[1]),
                            "inner_train_neg": int(train_counts[0]),
                            "inner_val_pos": int(val_counts[1]),
                            "inner_val_neg": int(val_counts[0]),
                            "inner_train_indices": ";".join(map(str, outer_train_idx[inner_train_local])),
                            "inner_val_indices": ";".join(map(str, outer_train_idx[inner_val_local])),
                        }
                    )
                    inner_split_defs.append((inner_cv_fold, inner_train_local, inner_val_local))
            else:
                raise ValueError(args.inner_selection)

            print(
                f"{dataset_key} | {input_set} | outer fold {outer_fold}: "
                f"inner {args.inner_selection} folds={len(inner_split_defs)}",
                flush=True,
            )

            for base_config_id, config in enumerate(base_configs, 1):
                started = time.time()
                for inner_cv_fold, inner_train_local, inner_val_local in inner_split_defs:
                    X_inner_train = X_outer_train[inner_train_local]
                    y_inner_train = y_outer_train[inner_train_local]
                    X_inner_val = X_outer_train[inner_val_local]
                    y_inner_val = y_outer_train[inner_val_local]
                    train_seed = (
                        args.seed
                        + 1_000_000 * outer_fold
                        + 10_000 * base_config_id
                        + 101 * inner_cv_fold
                        + group_id
                    )
                    checkpoint_rows = train_with_epoch_checkpoints(
                        config,
                        epochs,
                        X_inner_train,
                        y_inner_train,
                        X_inner_val,
                        y_inner_val,
                        seed=train_seed,
                        device=device,
                        sampler_multiplier=args.sampler_multiplier,
                        num_workers=args.num_workers,
                        loss_function=args.loss_function,
                        gce_q=args.gce_q,
                    )
                    for checkpoint_row in checkpoint_rows:
                        row = {
                            "dataset": dataset_key,
                            "dataset_title": dataset_spec["title"],
                            "input_set": input_set,
                            "outer_fold": outer_fold,
                            "base_config_id": base_config_id,
                            "n_features": X_all.shape[2],
                            "loss_function": args.loss_function,
                            "gce_q": args.gce_q if args.loss_function == "gce" else np.nan,
                            "inner_selection": args.inner_selection,
                            "inner_cv_fold": inner_cv_fold,
                            "inner_train_n": len(y_inner_train),
                            "inner_val_n": len(y_inner_val),
                            "fit_seconds_to_max_epoch": time.time() - started,
                            **asdict(config),
                            **checkpoint_row,
                        }
                        fold_inner_rows.append(row)

                if (
                    base_config_id == 1
                    or base_config_id % args.progress_every == 0
                    or base_config_id == len(base_configs)
                ):
                    print(
                        f"{dataset_key} | {input_set} | outer fold {outer_fold}: "
                        f"finished base config {base_config_id}/{len(base_configs)}",
                        flush=True,
                    )

            fold_inner_df = pd.DataFrame(fold_inner_rows)
            best = select_best_hparams(
                fold_inner_df,
                args.selection_metric,
                std_penalty=args.selection_std_penalty,
            )
            selected_by_fold[outer_fold] = best.copy()
            inner_rows.extend(fold_inner_rows)
            inner_train_n = int(round(float(best["inner_train_n"])))
            inner_val_n = int(round(float(best["inner_val_n"])))
            selected_from_input_set = input_set
        else:
            best = preselected_by_fold[outer_fold]
            inner_train_n = int(best["inner_train_n"])
            inner_val_n = int(best["inner_val_n"])
            selected_from_input_set = str(best["input_set"])
            split_rows.append(
                {
                    "dataset": dataset_key,
                    "dataset_title": dataset_spec["title"],
                    "input_set": input_set,
                    "outer_fold": outer_fold,
                    "outer_train_n": len(outer_train_idx),
                    "outer_val_n": len(outer_val_idx),
                    "hparam_scope": args.hparam_scope,
                    "selected_from_input_set": selected_from_input_set,
                    "valid_split": True,
                    "warning": "Inner search skipped; hyperparameters reused from sim.",
                }
            )
            print(
                f"{dataset_key} | {input_set} | outer fold {outer_fold}: "
                f"reusing hyperparameters from {selected_from_input_set}",
                flush=True,
            )
        selected_config = BaseConfig(
            embedding_dim=int(best["embedding_dim"]),
            hidden_dim=int(best["hidden_dim"]),
            lr=float(best["lr"]),
            batch_size=int(best["batch_size"]),
            dropout=float(best["dropout"]),
            weight_decay=float(best["weight_decay"]),
            num_layers=int(best["num_layers"]),
            init_mode=str(best["init_mode"]),
        )
        selected_epochs = int(best["epochs"])

        final_seed = args.seed + 2_000_000 * outer_fold + group_id
        if args.no_refit_on_outer_train:
            if preselected_by_fold is not None:
                raise ValueError(
                    "--no-refit-on-outer-train is not supported with "
                    "--hparam-scope dataset_sim for reused input sets."
                )
            final_X_train = X_inner_train
            final_y_train = y_inner_train
        else:
            final_X_train = X_outer_train
            final_y_train = y_outer_train

        model = fit_model(
            selected_config,
            final_X_train,
            final_y_train,
            epochs=selected_epochs,
            seed=final_seed,
            device=device,
            sampler_multiplier=args.sampler_multiplier,
            num_workers=args.num_workers,
            loss_function=args.loss_function,
            gce_q=args.gce_q,
        )
        outer_scores = predict_scores(model, X_outer_val, device)
        final_train_scores = predict_scores(model, final_X_train, device)
        outer_metrics = safe_metrics(y_outer_val, outer_scores)
        final_train_metrics = safe_metrics(final_y_train, final_train_scores)

        outer_row = {
            "dataset": dataset_key,
            "dataset_title": dataset_spec["title"],
            "input_set": input_set,
            "outer_fold": outer_fold,
            "outer_train_n": len(outer_train_idx),
            "outer_val_n": len(outer_val_idx),
            "inner_train_n": inner_train_n,
            "inner_val_n": inner_val_n,
            "loss_function": args.loss_function,
            "gce_q": args.gce_q if args.loss_function == "gce" else np.nan,
            "selection_metric": args.selection_metric,
            "selection_std_penalty": args.selection_std_penalty,
            "inner_selection": str(best.get("inner_selection", args.inner_selection)),
            "inner_cv_folds": int(best.get("inner_cv_folds", 1)),
            "hparam_scope": args.hparam_scope,
            "selected_from_input_set": selected_from_input_set,
            "refit_on_outer_train": not args.no_refit_on_outer_train,
            "selected_base_config_id": int(best["base_config_id"]),
            "selected_epochs": selected_epochs,
            "selected_embedding_dim": selected_config.embedding_dim,
            "selected_hidden_dim": selected_config.hidden_dim,
            "selected_lr": selected_config.lr,
            "selected_batch_size": selected_config.batch_size,
            "selected_dropout": selected_config.dropout,
            "selected_weight_decay": selected_config.weight_decay,
            "selected_num_layers": selected_config.num_layers,
            "selected_init_mode": selected_config.init_mode,
            "selected_inner_val_auc": float(best["inner_val_auc"]),
            "selected_inner_val_auc_sd": float(best.get("inner_val_auc_sd", 0.0)),
            "selected_inner_val_loss": float(best["inner_val_loss"]),
            "selected_inner_val_loss_sd": float(best.get("inner_val_loss_sd", 0.0)),
            "selected_inner_selection_score": float(
                best.get("inner_selection_score", best["inner_val_auc"])
            ),
            "selected_inner_train_auc": float(best["inner_train_auc"]),
            "selected_inner_train_loss": float(best["inner_train_loss"]),
            "outer_val_auc": outer_metrics["auc"],
            "outer_val_loss": outer_metrics["loss"],
            "outer_val_accuracy": outer_metrics["accuracy"],
            "final_train_auc": final_train_metrics["auc"],
            "final_train_loss": final_train_metrics["loss"],
            "final_train_accuracy": final_train_metrics["accuracy"],
        }
        outer_rows.append(outer_row)

        for patient_index, label, score in zip(outer_val_idx, y_outer_val, outer_scores):
            prediction_rows.append(
                {
                    "dataset": dataset_key,
                    "dataset_title": dataset_spec["title"],
                    "input_set": input_set,
                    "outer_fold": outer_fold,
                    "patient_index": int(patient_index),
                    "y_true": int(label),
                    "score": float(score),
                }
            )

        print(
            f"{dataset_key} | {input_set} | outer fold {outer_fold}: "
            f"selected emb={selected_config.embedding_dim}, hidden={selected_config.hidden_dim}, "
            f"lr={selected_config.lr:g}, batch={selected_config.batch_size}, "
            f"epochs={selected_epochs}, outer AUC={outer_metrics['auc']:.4f}",
            flush=True,
        )

    return inner_rows, outer_rows, prediction_rows, split_rows, selected_by_fold


def summarize_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, input_set), part in predictions_df.groupby(["dataset", "input_set"], sort=False):
        part = part.sort_values("patient_index")
        y = part["y_true"].to_numpy(dtype=int)
        scores = part["score"].to_numpy(dtype=float)
        ci = auc_ci_from_delong(y, scores)
        rows.append(
            {
                "dataset": dataset,
                "dataset_title": part["dataset_title"].iloc[0],
                "input_set": input_set,
                "n": len(part),
                "n_positive": int(y.sum()),
                "n_negative": int((1 - y).sum()),
                **ci,
            }
        )

    final_df = pd.DataFrame(rows)
    delong_rows = []
    for dataset, part in predictions_df.groupby("dataset", sort=False):
        baseline = part[part["input_set"] == "sim"].sort_values("patient_index")
        if baseline.empty:
            for input_set in part["input_set"].drop_duplicates():
                delong_rows.append(
                    {
                        "dataset": dataset,
                        "input_set": input_set,
                        "auc_delta_vs_sim": np.nan,
                        "delong_z_vs_sim": np.nan,
                        "delong_p_vs_sim": np.nan,
                    }
                )
            continue
        y_base = baseline["y_true"].to_numpy(dtype=int)
        baseline_scores = baseline["score"].to_numpy(dtype=float)
        for input_set, candidate in part.groupby("input_set", sort=False):
            candidate = candidate.sort_values("patient_index")
            y_candidate = candidate["y_true"].to_numpy(dtype=int)
            if not np.array_equal(y_base, y_candidate):
                raise RuntimeError(f"Label order mismatch for {dataset} {input_set} vs sim.")
            candidate_scores = candidate["score"].to_numpy(dtype=float)
            if input_set == "sim":
                stats = {
                    "auc_delta_vs_sim": 0.0,
                    "delong_z_vs_sim": 0.0,
                    "delong_p_vs_sim": 1.0,
                }
            else:
                stats = delong_test_against_baseline(y_base, baseline_scores, candidate_scores)
            delong_rows.append({"dataset": dataset, "input_set": input_set, **stats})

    delong_df = pd.DataFrame(delong_rows)
    return final_df.merge(delong_df, on=["dataset", "input_set"], how="left")


def add_selection_summary(final_df: pd.DataFrame, outer_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, input_set), part in outer_df.groupby(["dataset", "input_set"], sort=False):
        row = {"dataset": dataset, "input_set": input_set}
        for col in [
            "selected_epochs",
            "selected_embedding_dim",
            "selected_hidden_dim",
            "selected_lr",
            "selected_batch_size",
        ]:
            row[f"{col}_by_fold"] = ";".join(map(str, part[col].tolist()))
            row[f"{col}_mode"] = part[col].mode(dropna=False).iloc[0]
        row["mean_outer_fold_auc"] = part["outer_val_auc"].mean()
        row["sd_outer_fold_auc"] = part["outer_val_auc"].std()
        rows.append(row)
    selection_df = pd.DataFrame(rows)
    return final_df.merge(selection_df, on=["dataset", "input_set"], how="left")


def json_ready_args(args: argparse.Namespace) -> dict:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def write_excel(
    excel_path: Path,
    sheets: dict[str, pd.DataFrame],
) -> None:
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            if len(df) > 1_048_575:
                continue
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        workbook = writer.book
        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                sampled_cells = list(column_cells[: min(len(column_cells), 250)])
                values = [
                    str(cell.value) if cell.value is not None else ""
                    for cell in sampled_cells
                ]
                width = min(max(len(value) for value in values) + 2, 60)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width


def save_outputs(
    args: argparse.Namespace,
    final_df: pd.DataFrame,
    outer_df: pd.DataFrame,
    inner_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    splits_df: pd.DataFrame,
) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.out_dir / "final_results.csv", index=False)
    outer_df.to_csv(args.out_dir / "outer_fold_results.csv", index=False)
    inner_df.to_csv(args.out_dir / "inner_search_results.csv", index=False)
    predictions_df.to_csv(args.out_dir / "pooled_predictions.csv", index=False)
    splits_df.to_csv(args.out_dir / "validation_splits.csv", index=False)

    run_config = json_ready_args(args)
    run_config["created_by"] = Path(__file__).name
    with (args.out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    run_config_df = pd.DataFrame([run_config])
    sheets = {
        "final_results": final_df,
        "outer_fold_results": outer_df,
        "pooled_predictions": predictions_df,
        "validation_splits": splits_df,
        "run_config": run_config_df,
    }
    if not args.no_inner_search_sheet:
        sheets["inner_search_results"] = inner_df
    write_excel(args.out_dir / "nested_validation_results.xlsx", sheets)


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.out_dir = args.out_dir.resolve()
    device = get_device(args.device)
    if args.loss_function == "gce" and args.gce_q <= 0:
        raise ValueError("--gce-q must be positive for generalized cross-entropy loss.")
    epochs = parse_csv_numbers(args.epochs, int)
    if any(epoch < 1 for epoch in epochs):
        raise ValueError("epochs must be positive.")
    epochs = sorted(set(epochs))
    base_configs = build_base_grid(args)
    inner_row_multiplier = args.inner_cv_folds if args.inner_selection == "cv" else 1

    print(f"Device: {device}", flush=True)
    print(f"Datasets: {', '.join(args.datasets)}", flush=True)
    print(f"Input sets: {', '.join(args.input_sets)}", flush=True)
    print(f"Base configs per inner split: {len(base_configs)}", flush=True)
    print(f"Epoch checkpoints: {epochs}", flush=True)
    print(f"Loss function: {args.loss_function}", flush=True)
    if args.loss_function == "gce":
        print(f"GCE q: {args.gce_q}", flush=True)
    print(
        f"Inner rows per dataset/input/fold: "
        f"{len(base_configs) * len(epochs) * inner_row_multiplier}",
        flush=True,
    )
    print(f"Inner selection: {args.inner_selection}", flush=True)
    print(f"Inner CV folds: {args.inner_cv_folds}", flush=True)
    print(f"Selection std penalty: {args.selection_std_penalty}", flush=True)
    print(f"Hyperparameter scope: {args.hparam_scope}", flush=True)
    print(f"Output directory: {args.out_dir}", flush=True)

    all_inner_rows = []
    all_outer_rows = []
    all_prediction_rows = []
    all_split_rows = []

    started = time.time()
    if args.hparam_scope == "dataset_sim" and "sim" not in args.input_sets:
        raise ValueError("--hparam-scope dataset_sim requires sim in --input-sets.")

    for dataset_key in args.datasets:
        if args.hparam_scope == "dataset_sim":
            ordered_input_sets = ["sim"] + [
                input_set for input_set in args.input_sets if input_set != "sim"
            ]
            preselected_by_fold = None
            for input_set in ordered_input_sets:
                inner_rows, outer_rows, prediction_rows, split_rows, selected_by_fold = (
                    run_one_dataset_input(
                        dataset_key,
                        input_set,
                        args,
                        base_configs,
                        epochs,
                        device,
                        preselected_by_fold=preselected_by_fold,
                    )
                )
                if input_set == "sim":
                    preselected_by_fold = selected_by_fold
                all_inner_rows.extend(inner_rows)
                all_outer_rows.extend(outer_rows)
                all_prediction_rows.extend(prediction_rows)
                all_split_rows.extend(split_rows)
        else:
            for input_set in args.input_sets:
                inner_rows, outer_rows, prediction_rows, split_rows, _selected_by_fold = (
                    run_one_dataset_input(
                        dataset_key,
                        input_set,
                        args,
                        base_configs,
                        epochs,
                        device,
                    )
                )
                all_inner_rows.extend(inner_rows)
                all_outer_rows.extend(outer_rows)
                all_prediction_rows.extend(prediction_rows)
                all_split_rows.extend(split_rows)

    inner_df = pd.DataFrame(all_inner_rows)
    outer_df = pd.DataFrame(all_outer_rows)
    predictions_df = pd.DataFrame(all_prediction_rows)
    splits_df = pd.DataFrame(all_split_rows)
    final_df = summarize_predictions(predictions_df)
    final_df = add_selection_summary(final_df, outer_df)
    save_outputs(args, final_df, outer_df, inner_df, predictions_df, splits_df)

    cols = [
        "dataset_title",
        "input_set",
        "auc",
        "auc_ci95_low",
        "auc_ci95_high",
        "auc_delta_vs_sim",
        "delong_z_vs_sim",
        "delong_p_vs_sim",
    ]
    print("\nFinal pooled outer-fold results:", flush=True)
    print(final_df[cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nElapsed minutes: {(time.time() - started) / 60.0:.2f}", flush=True)
    print(f"Saved Excel: {args.out_dir / 'nested_validation_results.xlsx'}", flush=True)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
