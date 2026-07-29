import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import roc_curve, roc_auc_score
from MLstatkit import Delong_test  # <-- DeLong API

from model.kfold import SelectedDVHDataset
from model.train2 import train_model

device = "cpu"
# fx only
# 400 for gu>=1, 250 for gu>=2, 300 for gi
# sim + fx, 500 for gu, 300 for gi
num_epochs = 300
k = 5
# sim + fx, 32
# fix only 64
batch_size = 32
downsample_rate = 12
num_repeats = 5
baseline = "SIM"

import seaborn as sns
sns.set_theme(style='whitegrid')

# Define your model configs: (name, tx_idx list)
model_cfgs = [
    ("SIM",        [0]),            # <-- baseline to compare against
    ("Fx1-5",      [0, 1, 2, 3, 4, 5]),
    ("Fx1+Fx2",    [0, 1, 2]),
    ("Fx1+Fx2+Fx3",[0, 1, 2, 3]),
    ("Fx4+Fx5",    [0, 4, 5]),
]


def run_once_get_scores_labels_ids(dataset, batch_size, num_epochs, k, pred_gu=False):
    """
    Expect train_model to return either:
        (scores, labels)  OR  (scores, labels, ids)
    IDs help align samples if order differs between models.
    """
    out = train_model(dataset, batch_size, num_epochs, k, pred_gu=False)
    try:
        y_scores, y_true, ids = out
    except Exception:
        y_scores, y_true = out
        ids = None  # fallback: assume identical order across models

    y_scores = np.asarray(y_scores).reshape(-1)
    y_true   = np.asarray(y_true).reshape(-1).astype(int)
    if ids is not None:
        ids = np.asarray(ids).reshape(-1)
        assert len(y_scores) == len(y_true) == len(ids)
    else:
        assert len(y_scores) == len(y_true)
    return y_scores, y_true, ids

# Concatenated containers
labels_all = []                       # concatenated gts across repeats
scores_all = {n: [] for n, _ in model_cfgs}  # concatenated scores per model

for r in range(num_repeats):
    print(f"\n=== Repeat {r+1}/{num_repeats} ===")
    per_model = {}  # name -> (scores, labels, ids)

    # 1) Run all models for this repeat
    for name, idx in model_cfgs:
        # if name.upper() == "SIM":
        #     num_epochs = 500
        #     batch_size = 32
        # else:
        #     num_epochs = 250
        #     batch_size = 64

        dataset_gu = SelectedDVHDataset(downsample_rate, pred_gu=False, tx_idx=idx)
        s, y, ids = run_once_get_scores_labels_ids(dataset_gu, batch_size, num_epochs, k, pred_gu=False)
        per_model[name] = (s, y, ids)

    # 2) Choose baseline as reference, align others to its order (if IDs available)
    base_scores, base_labels, base_ids = per_model[baseline]

    # If IDs exist, align everyone to baseline by IDs; else require same order & labels
    if base_ids is not None:
        # build index map for baseline
        base_map = {int(i): j for j, i in enumerate(base_ids)}
        def align_to_baseline(scores, labels, ids):
            # reorder to match baseline order
            order = [base_map[int(i)] for i in ids]
            # We assume same ID set; if not, raise
            if len(order) != len(base_ids):
                raise RuntimeError("ID set mismatch across models in repeat.")
            return scores[order], labels[order]
        # align everyone
        aligned = {}
        for name, (s, y, ids) in per_model.items():
            if ids is None:
                raise RuntimeError("Baseline returned IDs but another model did not; please return IDs for all.")
            s2, y2 = align_to_baseline(s, y, ids)
            aligned[name] = (s2, y2)
    else:
        # no IDs: enforce identical labels/order across models
        aligned = {}
        for name, (s, y, ids) in per_model.items():
            if not np.array_equal(y, base_labels):
                raise RuntimeError("Labels/order differ across models; return IDs from train_model to allow alignment.")
            aligned[name] = (s, y)

    # 3) Concatenate this repeat’s baseline labels & each model’s scores
    #    (labels are the same across models after alignment)
    labels_all.append(aligned[baseline][1])  # same for all models after alignment
    for name in scores_all:
        scores_all[name].append(aligned[name][0])

# Stack all repeats end-to-end
labels_all = np.concatenate(labels_all, axis=0)
for name in scores_all:
    scores_all[name] = np.concatenate(scores_all[name], axis=0)

# --- Plot ROC from concatenated predictions --------------------------------
# plt.figure()
# for name, scores in scores_all.items():
#     fpr, tpr, _ = roc_curve(labels_all, scores)
#     auc = roc_auc_score(labels_all, scores)
#     plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
# plt.plot([0, 1], [0, 1], "--", color="gray")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title(f"ROC (concatenated over {num_repeats} repeats)")
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()

print("\nDeLong test on concatenated predictions (each vs SIM):")
for name, scores in scores_all.items():
    if name == baseline:
        continue
    z, p = Delong_test(labels_all, scores_all[baseline], scores)
    print(f"{name} vs {baseline}: z={z:.3f}, p={p:.6g}")