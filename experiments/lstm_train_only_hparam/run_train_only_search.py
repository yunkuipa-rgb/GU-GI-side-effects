#!/usr/bin/env python3
"""LSTM hyperparameter search without using held-out CV AUC for selection.

For each candidate configuration, features are selected inside each outer
training fold only. The final per-task configuration is selected by a
training-only criterion, then its held-out 5-fold AUC is reported.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import sem
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from model import LSTMClassifier


TASKS = {
    "GU1": {
        "file": "selected_dvh_curves_gu_1.npz",
        "label": "GU",
        "title": "GU >= 1",
    },
    "GU2": {
        "file": "selected_dvh_curves_gu_2.npz",
        "label": "GU",
        "title": "GU >= 2",
    },
    "GI": {
        "file": "selected_dvh_curves_gi.npz",
        "label": "GI",
        "title": "GI >= 1",
    },
}

FRACTION_SETS = {
    "SIM": [0],
    "SIM+Fx1": [0, 1],
    "SIM+Fx1+Fx2": [0, 1, 2],
    "SIM+Fx1+Fx2+Fx3": [0, 1, 2, 3],
    "SIM+Fx1-Fx4": [0, 1, 2, 3, 4],
    "SIM+Fx1-Fx5": [0, 1, 2, 3, 4, 5],
}


@dataclass(frozen=True)
class Config:
    fraction_set: str
    feature_count: int
    hidden_dim: int
    lr: float
    epochs: int
    batch_size: int
    dropout: float
    weight_decay: float
    num_layers: int
    init_mode: str

    @property
    def embedding_dim(self) -> int:
        return self.hidden_dim


def parse_csv_numbers(value: str, caster=float) -> list:
    return [caster(item.strip()) for item in value.split(",") if item.strip()]


def parse_feature_counts(value: str) -> list[int]:
    counts: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = [int(x) for x in part.split("-", 1)]
            counts.extend(range(low, high + 1))
        else:
            counts.append(int(part))
    counts = sorted(set(counts))
    if any(count < 1 for count in counts):
        raise ValueError("Feature counts must be positive.")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training-only LSTM hyperparameter search."
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
        help="Directory for CSV and PNG outputs.",
    )
    parser.add_argument("--tasks", nargs="+", default=["GU1", "GU2", "GI"], choices=TASKS)
    parser.add_argument(
        "--fraction-sets",
        nargs="+",
        default=["SIM", "SIM+Fx1", "SIM+Fx1+Fx2", "SIM+Fx1+Fx2+Fx3", "SIM+Fx1-Fx4", "SIM+Fx1-Fx5"],
        choices=FRACTION_SETS,
    )
    parser.add_argument(
        "--feature-counts",
        default="5,10,15,20,25,30",
        help="Feature counts, e.g. '5,10,15,20,25,30' or full range '5-30'.",
    )
    parser.add_argument("--hidden-dims", default="64,128,192")
    parser.add_argument("--lrs", default="0.001,0.0005")
    parser.add_argument("--epochs", default="150")
    parser.add_argument("--batch-sizes", default="32")
    parser.add_argument("--dropouts", default="0.1")
    parser.add_argument("--weight-decays", default="1e-8")
    parser.add_argument("--num-layers", default="4")
    parser.add_argument("--init-mode", default="zeros", choices=["zeros", "random"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-metric",
        default="train_loss",
        choices=["train_loss", "train_auc", "train_loss_1se"],
        help="Training-only metric used to select the final config per task.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Optional cap for smoke tests; takes the first N grid configs.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--balance-mode",
        default="weighted_loss",
        choices=["weighted_loss", "sampler"],
        help="Class imbalance handling for LSTM training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

        arr = np.asarray(data[key], dtype=float) / 100.0
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
    collection = np.transpose(collection, (2, 0, 1))  # [patient, time point, feature]
    if collection.shape[0] != len(y):
        raise ValueError(f"{npz_path} has {collection.shape[0]} patients but {len(y)} labels.")
    return collection, y, feature_names


def select_fractions(collection: np.ndarray, fraction_set: str) -> np.ndarray:
    tx_idx = FRACTION_SETS[fraction_set]
    if max(tx_idx) >= collection.shape[1]:
        raise ValueError(
            f"{fraction_set} requests time index {max(tx_idx)} but only "
            f"{collection.shape[1]} time points are available."
        )
    return collection[:, tx_idx, :]


def select_features_from_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_count: int,
    agg: str = "max",
) -> np.ndarray:
    n_features = X_train.shape[2]
    if feature_count >= n_features:
        return np.arange(n_features)

    n_samples, n_time, _ = X_train.shape
    flat = X_train.reshape(n_samples, n_time * n_features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, _ = f_classif(flat, y_train)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = scores.reshape(n_time, n_features)

    if agg == "max":
        feature_scores = scores.max(axis=0)
    elif agg == "mean":
        feature_scores = scores.mean(axis=0)
    else:
        raise ValueError(f"Unknown feature score aggregation: {agg}")

    return np.argsort(-feature_scores)[:feature_count]


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    use_sampler: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32),
    )
    if not use_sampler:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    counts = np.bincount(y.astype(int), minlength=2)
    class_weights = np.zeros_like(counts, dtype=float)
    nonzero = counts > 0
    class_weights[nonzero] = 1.0 / counts[nonzero]
    sample_weights = class_weights[y.astype(int)]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights) * 2,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def predict_scores(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        scores = model(tensor).detach().cpu().numpy().reshape(-1)
    model.train()
    return scores


def metrics_for(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    scores = np.clip(scores, 1e-7, 1 - 1e-7)
    pred = (scores > 0.5).astype(int)
    auc = roc_auc_score(y, scores) if len(np.unique(y)) == 2 else np.nan
    return {
        "loss": log_loss(y, scores, labels=[0, 1]),
        "auc": auc,
        "accuracy": accuracy_score(y, pred),
    }


def train_one_fold(
    config: Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    device: torch.device,
    balance_mode: str,
) -> dict[str, float]:
    set_seed(seed)
    model = LSTMClassifier(
        n_features=X_train.shape[2],
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        output_size=1,
        num_layers=config.num_layers,
        dropout=config.dropout,
        init_mode=config.init_mode,
    ).to(device)

    loader = make_loader(
        X_train,
        y_train,
        batch_size=config.batch_size,
        use_sampler=True,
        seed=seed,
    )
    criterion = torch.nn.BCELoss()
    weighted_criterion = torch.nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    if balance_mode == "weighted_loss":
        batch_x = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        batch_y = torch.as_tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
        counts = np.bincount(y_train.astype(int), minlength=2)
        class_weights = np.zeros_like(counts, dtype=float)
        nonzero = counts > 0
        class_weights[nonzero] = len(y_train) / (2.0 * counts[nonzero])
        sample_weights = torch.as_tensor(
            class_weights[y_train.astype(int)].reshape(-1, 1),
            dtype=torch.float32,
            device=device,
        )
        for _ in range(config.epochs):
            optimizer.zero_grad()
            loss = weighted_criterion(model(batch_x), batch_y)
            loss = (loss * sample_weights).mean()
            loss.backward()
            optimizer.step()
    elif balance_mode == "sampler":
        for _ in range(config.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
    else:
        raise ValueError(f"Unknown balance_mode: {balance_mode}")

    train_scores = predict_scores(model, X_train, device)
    test_scores = predict_scores(model, X_test, device)
    train_metrics = metrics_for(y_train, train_scores)
    test_metrics = metrics_for(y_test, test_scores)

    return {
        "train_loss": train_metrics["loss"],
        "train_auc": train_metrics["auc"],
        "train_accuracy": train_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_auc": test_metrics["auc"],
        "test_accuracy": test_metrics["accuracy"],
    }


def build_grid(args: argparse.Namespace) -> list[Config]:
    configs = []
    for values in itertools.product(
        args.fraction_sets,
        parse_feature_counts(args.feature_counts),
        parse_csv_numbers(args.hidden_dims, int),
        parse_csv_numbers(args.lrs, float),
        parse_csv_numbers(args.epochs, int),
        parse_csv_numbers(args.batch_sizes, int),
        parse_csv_numbers(args.dropouts, float),
        parse_csv_numbers(args.weight_decays, float),
        parse_csv_numbers(args.num_layers, int),
    ):
        configs.append(Config(*values, init_mode=args.init_mode))
    if args.max_configs is not None:
        return configs[: args.max_configs]
    return configs


def run_task(
    task_key: str,
    args: argparse.Namespace,
    configs: list[Config],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task = TASKS[task_key]
    collection, y, feature_names = load_collection(args.root / task["file"], task["label"])
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = list(splitter.split(collection, y))
    fold_rows = []

    for config_idx, config in enumerate(configs, 1):
        X_all = select_fractions(collection, config.fraction_set)
        for fold, (train_idx, test_idx) in enumerate(folds, 1):
            X_train_full = X_all[train_idx]
            y_train = y[train_idx]
            X_test_full = X_all[test_idx]
            y_test = y[test_idx]
            selected_idx = select_features_from_training(
                X_train_full,
                y_train,
                config.feature_count,
                agg="max",
            )
            X_train = X_train_full[:, :, selected_idx]
            X_test = X_test_full[:, :, selected_idx]
            fold_seed = args.seed + config_idx * 100 + fold
            metrics = train_one_fold(
                config,
                X_train,
                y_train,
                X_test,
                y_test,
                fold_seed,
                device,
                args.balance_mode,
            )

            row = {
                "task": task_key,
                "task_title": task["title"],
                "config_id": config_idx,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_available_features": X_all.shape[2],
                "n_selected_features": len(selected_idx),
                "selected_features": ";".join(feature_names[i] for i in selected_idx),
                "balance_mode": args.balance_mode,
                **asdict(config),
                "embedding_dim": config.embedding_dim,
                **metrics,
            }
            fold_rows.append(row)

        if config_idx == 1 or config_idx % 10 == 0 or config_idx == len(configs):
            print(f"{task_key}: finished config {config_idx}/{len(configs)}", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    group_cols = [
        "task",
        "task_title",
        "config_id",
        "fraction_set",
        "feature_count",
        "hidden_dim",
        "embedding_dim",
        "lr",
        "epochs",
        "batch_size",
        "dropout",
        "weight_decay",
        "num_layers",
        "init_mode",
    ]
    summary = (
        fold_df.groupby(group_cols, sort=False)
        .agg(
            mean_train_loss=("train_loss", "mean"),
            sd_train_loss=("train_loss", "std"),
            mean_train_auc=("train_auc", "mean"),
            mean_train_accuracy=("train_accuracy", "mean"),
            mean_test_auc=("test_auc", "mean"),
            sd_test_auc=("test_auc", "std"),
            mean_test_accuracy=("test_accuracy", "mean"),
            mean_test_loss=("test_loss", "mean"),
        )
        .reset_index()
    )
    summary["sem_train_loss"] = summary["sd_train_loss"] / math.sqrt(args.folds)
    summary["sem_test_auc"] = summary["sd_test_auc"] / math.sqrt(args.folds)
    summary["ci95_test_auc_low"] = summary["mean_test_auc"] - 1.96 * summary["sem_test_auc"]
    summary["ci95_test_auc_high"] = summary["mean_test_auc"] + 1.96 * summary["sem_test_auc"]
    return fold_df, summary


def select_training_only(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    fraction_count = {
        "SIM": 1,
        "SIM+Fx1": 2,
        "SIM+Fx1+Fx2": 3,
        "SIM+Fx1+Fx2+Fx3": 4,
        "SIM+Fx1-Fx4": 5,
        "SIM+Fx1-Fx5": 6,
    }
    selected_rows = []
    for task, part in summary.groupby("task", sort=False):
        if metric == "train_loss":
            ordered = part.sort_values(
                ["mean_train_loss", "mean_train_auc"],
                ascending=[True, False],
            )
        elif metric == "train_auc":
            ordered = part.sort_values(
                ["mean_train_auc", "mean_train_loss"],
                ascending=[False, True],
            )
        elif metric == "train_loss_1se":
            best = part.sort_values("mean_train_loss").iloc[0]
            threshold = best["mean_train_loss"] + best["sem_train_loss"]
            ordered = part[part["mean_train_loss"] <= threshold].copy()
            ordered["fraction_count"] = ordered["fraction_set"].map(fraction_count)
            ordered = ordered.sort_values(
                [
                    "hidden_dim",
                    "epochs",
                    "fraction_count",
                    "feature_count",
                    "mean_train_loss",
                ],
                ascending=[True, True, True, True, True],
            )
        else:
            raise ValueError(metric)
        row = ordered.iloc[0].copy()
        row["selection_metric"] = metric
        row["selection_note"] = "selected without using held-out fold AUC"
        if metric == "train_loss_1se":
            row["best_mean_train_loss"] = best["mean_train_loss"]
            row["train_loss_1se_threshold"] = threshold
        selected_rows.append(row)
    return pd.DataFrame(selected_rows)


def save_plot(summary: pd.DataFrame, selected: pd.DataFrame, out_path: Path) -> None:
    tasks = list(summary["task"].drop_duplicates())
    fig, axes = plt.subplots(len(tasks), 1, figsize=(9, 3.5 * len(tasks)), squeeze=False)

    for row, task in enumerate(tasks):
        ax = axes[row][0]
        part = summary[summary["task"] == task]
        for fraction_set, group in part.groupby("fraction_set", sort=False):
            best_by_feature = (
                group.groupby("feature_count")["mean_test_auc"].max().reset_index()
            )
            ax.plot(
                best_by_feature["feature_count"],
                best_by_feature["mean_test_auc"],
                marker="o",
                linewidth=1.2,
                label=fraction_set,
            )

        selected_task = selected[selected["task"] == task]
        if not selected_task.empty:
            row0 = selected_task.iloc[0]
            ax.scatter(
                [row0["feature_count"]],
                [row0["mean_test_auc"]],
                color="black",
                marker="x",
                s=80,
                label="training-selected config",
                zorder=5,
            )

        ax.axhline(0.5, color="0.65", linestyle="--", linewidth=1)
        ax.set_title(task)
        ax.set_xlabel("Selected feature count")
        ax.set_ylabel("Held-out 5-fold AUC")
        ax.set_ylim(0.2, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    configs = build_grid(args)
    print(f"Device: {device}", flush=True)
    print(f"Tasks: {', '.join(args.tasks)}", flush=True)
    print(f"Grid configs per task: {len(configs)}", flush=True)
    print(f"Selection metric: {args.selection_metric}", flush=True)

    fold_frames = []
    summary_frames = []
    for task in args.tasks:
        fold_df, summary_df = run_task(task, args, configs, device)
        fold_frames.append(fold_df)
        summary_frames.append(summary_df)

    fold_results = pd.concat(fold_frames, ignore_index=True)
    config_summary = pd.concat(summary_frames, ignore_index=True)
    selected = select_training_only(config_summary, args.selection_metric)

    fold_results.to_csv(args.out_dir / "fold_results.csv", index=False)
    config_summary.to_csv(args.out_dir / "config_summary.csv", index=False)
    selected.to_csv(args.out_dir / "selected_by_training.csv", index=False)
    save_plot(config_summary, selected, args.out_dir / "auc_by_feature_count.png")

    cols = [
        "task",
        "fraction_set",
        "feature_count",
        "hidden_dim",
        "lr",
        "epochs",
        "batch_size",
        "dropout",
        "mean_train_loss",
        "mean_train_auc",
        "mean_test_auc",
        "ci95_test_auc_low",
        "ci95_test_auc_high",
    ]
    print("\nSelected by training-only criterion:")
    print(selected[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
