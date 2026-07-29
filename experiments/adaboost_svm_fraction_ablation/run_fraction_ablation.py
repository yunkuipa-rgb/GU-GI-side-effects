#!/usr/bin/env python3
"""AdaBoost/SVM fraction ablation on the selected DVH feature tensors.

The default experiment mirrors the LSTM input representation:
SIM/Fx time points are selected from the existing selected_dvh_curves_*.npz
files, flattened, and evaluated with classical classifiers.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import VarianceThreshold


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
        "title": "GI",
    },
}

FRACTION_SETS = {
    "SIM": [0],
    "Fx1": [1],
    "SIM+Fx1": [0, 1],
    "Fx1+Fx2": [1, 2],
    "SIM+Fx1+Fx2": [0, 1, 2],
    "Fx1+Fx2+Fx3": [1, 2, 3],
    "SIM+Fx1+Fx2+Fx3": [0, 1, 2, 3],
    "Fx1-Fx4": [1, 2, 3, 4],
    "SIM+Fx1-Fx4": [0, 1, 2, 3, 4],
    "Fx1-Fx5": [1, 2, 3, 4, 5],
    "SIM+Fx1-Fx5": [0, 1, 2, 3, 4, 5],
    "Fx4+Fx5": [4, 5],
    "SIM+Fx4+Fx5": [0, 4, 5],
}


class SelectKBestIfPossible(BaseEstimator, TransformerMixin):
    """Use SelectKBest only when the requested k is smaller than n_features."""

    def __init__(self, k: int | None = 30):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        if self.k is None or self.k >= X.shape[1]:
            self.selector_ = None
            self.n_features_out_ = X.shape[1]
            return self

        self.selector_ = SelectKBest(score_func=f_classif, k=self.k)
        self.selector_.fit(X, y)
        self.n_features_out_ = self.k
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selector_ is None:
            return X
        return self.selector_.transform(X)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SIM/fraction subsets with AdaBoost and linear SVM."
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
        "--models", nargs="+", default=["adaboost", "svm"], choices=["adaboost", "svm"]
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--select-k",
        default="none",
        help="Optional ANOVA features retained inside each training fold. Use 'none' to disable.",
    )
    parser.add_argument("--ada-estimators", type=int, default=200)
    parser.add_argument("--ada-learning-rate", type=float, default=1.0)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument(
        "--reference",
        default="SIM+Fx1+Fx2",
        choices=list(FRACTION_SETS),
        help="Fraction set used for paired delta columns.",
    )
    return parser.parse_args()


def selected_k(value: str) -> int | None:
    if value.lower() in {"none", "no", "false", "all"}:
        return None
    k = int(value)
    if k <= 0:
        raise ValueError("--select-k must be a positive integer or 'none'.")
    return k


def load_collection(npz_path: Path, label_key: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    y = np.asarray(data[label_key]).reshape(-1).astype(int)

    pieces = []
    for key in data.files:
        if key in {"GU", "GI"}:
            continue

        arr = np.asarray(data[key], dtype=float) / 100.0
        if arr.ndim != 2:
            raise ValueError(f"{npz_path}:{key} must be 2D, got {arr.shape}")

        if arr.shape[0] == 1:
            pieces.append(np.repeat(arr, 6, axis=0))
        elif arr.shape[0] == 6:
            pieces.append(arr)
        elif arr.shape[0] == 12:
            pieces.append(arr[0::2, :])
            pieces.append(arr[1::2, :])
        else:
            raise ValueError(
                f"{npz_path}:{key} has {arr.shape[0]} time points; expected 1, 6, or 12."
            )

    collection = np.stack(pieces, axis=1)  # [time point, feature, patient]
    if collection.shape[-1] != len(y):
        raise ValueError(
            f"{npz_path} has {collection.shape[-1]} patients but {len(y)} labels."
        )
    return collection, y


def flatten_fraction_set(collection: np.ndarray, tx_idx: list[int]) -> np.ndarray:
    if max(tx_idx) >= collection.shape[0]:
        raise ValueError(
            f"Requested fraction index {max(tx_idx)} but collection has "
            f"{collection.shape[0]} time points."
        )
    selected = collection[tx_idx, :, :]
    return np.transpose(selected, (2, 0, 1)).reshape(collection.shape[-1], -1)


def build_estimator(
    model_name: str,
    seed: int,
    select_k_value: int | None,
    ada_estimators: int,
    ada_learning_rate: float,
    svm_c: float,
) -> Pipeline:
    steps = [("variance", VarianceThreshold())]
    if select_k_value is not None:
        steps.append(("select", SelectKBestIfPossible(k=select_k_value)))

    if model_name == "adaboost":
        classifier = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=seed),
            n_estimators=ada_estimators,
            learning_rate=ada_learning_rate,
            random_state=seed,
        )
    elif model_name == "svm":
        steps.append(("scale", StandardScaler()))
        classifier = SVC(kernel="linear", C=svm_c, class_weight="balanced")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    steps.append(("model", classifier))
    return Pipeline(steps)


def binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def decision_scores(estimator: Pipeline, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    select_k_value = selected_k(args.select_k)
    metric_rows = []
    prediction_rows = []

    for task_key in args.tasks:
        task = TASKS[task_key]
        collection, y = load_collection(args.root / task["file"], task["label"])

        for fraction_name, tx_idx in FRACTION_SETS.items():
            X = flatten_fraction_set(collection, tx_idx)

            for model_name in args.models:
                for repeat in range(args.repeats):
                    splitter = StratifiedKFold(
                        n_splits=args.folds,
                        shuffle=True,
                        random_state=args.seed + repeat,
                    )
                    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), 1):
                        estimator_seed = args.seed + repeat * 100 + fold
                        estimator = build_estimator(
                            model_name,
                            estimator_seed,
                            select_k_value,
                            args.ada_estimators,
                            args.ada_learning_rate,
                            args.svm_c,
                        )
                        fit_kwargs = {}
                        if model_name == "adaboost":
                            fit_kwargs["model__sample_weight"] = compute_sample_weight(
                                class_weight="balanced", y=y[train_idx]
                            )

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            estimator.fit(X[train_idx], y[train_idx], **fit_kwargs)

                        y_score = decision_scores(estimator, X[test_idx])
                        y_pred = estimator.predict(X[test_idx])
                        metrics = binary_metrics(y[test_idx], y_pred, y_score)
                        metric_rows.append(
                            {
                                "task": task_key,
                                "task_title": task["title"],
                                "model": model_name,
                                "fraction_set": fraction_name,
                                "repeat": repeat + 1,
                                "fold": fold,
                                "n_train": len(train_idx),
                                "n_test": len(test_idx),
                                "n_features_raw": X.shape[1],
                                **metrics,
                            }
                        )

                        for patient_idx, truth, score, pred in zip(
                            test_idx, y[test_idx], y_score, y_pred
                        ):
                            prediction_rows.append(
                                {
                                    "task": task_key,
                                    "model": model_name,
                                    "fraction_set": fraction_name,
                                    "repeat": repeat + 1,
                                    "fold": fold,
                                    "patient_id": int(patient_idx) + 1,
                                    "y_true": int(truth),
                                    "y_score": float(score),
                                    "y_pred": int(pred),
                                }
                            )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def summarize_metrics(metrics: pd.DataFrame, reference: str) -> pd.DataFrame:
    group_cols = ["task", "task_title", "model", "fraction_set"]
    summary = (
        metrics.groupby(group_cols, sort=False)
        .agg(
            n_splits=("auc", "count"),
            n_features_raw=("n_features_raw", "first"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
        )
        .reset_index()
    )
    summary["auc_sem"] = summary["auc_std"] / np.sqrt(summary["n_splits"])
    summary["auc_ci95_low"] = summary["auc_mean"] - 1.96 * summary["auc_sem"]
    summary["auc_ci95_high"] = summary["auc_mean"] + 1.96 * summary["auc_sem"]

    paired_cols = ["task", "model", "repeat", "fold"]
    ref = metrics[metrics["fraction_set"] == reference][paired_cols + ["auc"]]
    ref = ref.rename(columns={"auc": "reference_auc"})
    paired = metrics.merge(ref, on=paired_cols, how="left")
    paired["delta_auc_vs_reference"] = paired["auc"] - paired["reference_auc"]

    delta_rows = []
    for keys, part in paired.groupby(group_cols, sort=False):
        delta = part["delta_auc_vs_reference"].dropna()
        p_value = np.nan
        if len(delta) > 0 and not np.allclose(delta.to_numpy(), 0):
            try:
                p_value = wilcoxon(delta).pvalue
            except ValueError:
                p_value = np.nan
        delta_rows.append(
            {
                **dict(zip(group_cols, keys)),
                "delta_auc_vs_reference_mean": delta.mean(),
                "delta_auc_vs_reference_std": delta.std(),
                "wilcoxon_p_vs_reference": p_value,
            }
        )

    deltas = pd.DataFrame(delta_rows)
    return summary.merge(deltas, on=group_cols, how="left")


def save_plot(summary: pd.DataFrame, out_path: Path) -> None:
    tasks = list(summary["task"].drop_duplicates())
    models = list(summary["model"].drop_duplicates())
    fig, axes = plt.subplots(
        len(tasks),
        len(models),
        figsize=(5.0 * len(models), 3.8 * len(tasks)),
        squeeze=False,
        sharey=True,
    )

    order = list(FRACTION_SETS)
    x = np.arange(len(order))
    for row, task in enumerate(tasks):
        for col, model in enumerate(models):
            ax = axes[row][col]
            part = summary[(summary["task"] == task) & (summary["model"] == model)]
            part = part.set_index("fraction_set").reindex(order)
            y = part["auc_mean"].to_numpy()
            yerr = (
                1.96
                * part["auc_std"].to_numpy()
                / np.sqrt(part["n_splits"].to_numpy())
            )
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.5)
            ax.axhline(0.5, color="0.65", linestyle="--", linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(order, rotation=35, ha="right")
            ax.set_ylim(0.2, 1.0)
            ax.set_title(f"{task} / {model}")
            ax.set_ylabel("Mean fold AUC")
            ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics, predictions = run_experiment(args)
    summary = summarize_metrics(metrics, args.reference)

    metrics_path = args.out_dir / "fold_metrics.csv"
    predictions_path = args.out_dir / "pooled_predictions.csv"
    summary_path = args.out_dir / "summary.csv"
    plot_path = args.out_dir / "auc_by_fraction.png"

    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_plot(summary, plot_path)

    display_cols = [
        "task",
        "model",
        "fraction_set",
        "n_features_raw",
        "auc_mean",
        "auc_ci95_low",
        "auc_ci95_high",
        "delta_auc_vs_reference_mean",
        "wilcoxon_p_vs_reference",
    ]
    print(summary[display_cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {predictions_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
