import os
import glob
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models
from torchvision.models import ResNet50_Weights

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def list_nifti_files(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "**", "*.nii"), recursive=True))
    files += sorted(glob.glob(os.path.join(input_dir, "**", "*.nii.gz"), recursive=True))
    return sorted(list(set(files)))


def robust_minmax(slice_2d, lower=1.0, upper=99.0, eps=1e-8):
    """
    Percentile-based normalization for one 2D slice.
    Output range is [0, 1].
    """
    lo = np.percentile(slice_2d, lower)
    hi = np.percentile(slice_2d, upper)
    x = np.clip(slice_2d, lo, hi)
    x = (x - lo) / (hi - lo + eps)
    return x.astype(np.float32)


def center_crop_or_pad(img, out_h, out_w):
    """
    img: [H, W]
    Returns [out_h, out_w]
    """
    h, w = img.shape
    out = np.zeros((out_h, out_w), dtype=img.dtype)

    src_y0 = max((h - out_h) // 2, 0)
    src_x0 = max((w - out_w) // 2, 0)
    src_y1 = min(src_y0 + out_h, h)
    src_x1 = min(src_x0 + out_w, w)

    dst_y0 = max((out_h - h) // 2, 0)
    dst_x0 = max((out_w - w) // 2, 0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return out


class NiftiSliceDataset(Dataset):
    def __init__(
        self,
        nifti_files,
        slice_axis=2,
        out_size=224,
        skip_blank=True,
        blank_threshold=1e-6,
        infer_label_from_parent=True,
    ):
        self.samples = []
        self.out_size = out_size
        self.slice_axis = slice_axis
        self.skip_blank = skip_blank
        self.blank_threshold = blank_threshold
        self.infer_label_from_parent = infer_label_from_parent

        for fp in nifti_files:
            img = nib.load(fp)
            vol = img.get_fdata()

            if vol.ndim != 3:
                print(f"[Skip] Not 3D: {fp}, shape={vol.shape}")
                continue

            n_slices = vol.shape[slice_axis]
            label = Path(fp).parent.name if infer_label_from_parent else "unknown"

            for s in range(n_slices):
                self.samples.append((fp, s, label))

        print(f"Collected {len(self.samples)} raw slices from {len(nifti_files)} volumes.")

    def __len__(self):
        return len(self.samples)

    def _extract_slice(self, vol, slice_idx):
        if self.slice_axis == 0:
            sl = vol[slice_idx, :, :]
        elif self.slice_axis == 1:
            sl = vol[:, slice_idx, :]
        elif self.slice_axis == 2:
            sl = vol[:, :, slice_idx]
        else:
            raise ValueError(f"Invalid slice_axis={self.slice_axis}")
        return sl

    def __getitem__(self, idx):
        fp, slice_idx, label = self.samples[idx]

        vol = nib.load(fp).get_fdata()
        sl = self._extract_slice(vol, slice_idx).astype(np.float32)

        # Optionally skip nearly blank slices by returning a flag
        is_blank = np.std(sl) < self.blank_threshold

        # Normalize
        sl = robust_minmax(sl)

        # Resize by crop/pad to 224x224
        sl = center_crop_or_pad(sl, self.out_size, self.out_size)

        # Convert grayscale to 3 channels
        sl = np.stack([sl, sl, sl], axis=0)  # [3, H, W]

        sample = {
            "image": torch.from_numpy(sl),       # float32, [3, 224, 224]
            "file_path": fp,
            "slice_idx": slice_idx,
            "label": label,
            "is_blank": is_blank,
        }
        return sample


def build_feature_extractor(device):
    """
    ResNet-50 pretrained on ImageNet.
    Remove the final classification layer so output is [B, 2048].
    """
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    # Use the official ImageNet normalization
    mean = torch.tensor(weights.transforms().mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(weights.transforms().std).view(1, 3, 1, 1).to(device)

    return model, mean, std


@torch.no_grad()
def extract_features(
    dataset,
    batch_size,
    num_workers,
    device,
    skip_blank_runtime=True
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model, mean, std = build_feature_extractor(device)

    all_features = []
    all_file_paths = []
    all_slice_indices = []
    all_labels = []

    for batch in loader:
        x = batch["image"].to(device)  # [B, 3, 224, 224]
        is_blank = batch["is_blank"].numpy().astype(bool)

        # Runtime skip blank slices
        keep = np.logical_not(is_blank) if skip_blank_runtime else np.ones_like(is_blank, dtype=bool)
        if keep.sum() == 0:
            continue

        keep_t = torch.from_numpy(keep).to(device)
        x = x[keep_t]

        # ImageNet normalization
        x = (x - mean) / std

        feat = model(x)  # [B_keep, 2048]
        feat = feat.cpu().numpy().astype(np.float32)

        kept_paths = [p for p, k in zip(batch["file_path"], keep) if k]
        kept_slices = [int(s) for s, k in zip(batch["slice_idx"], keep) if k]
        kept_labels = [lb for lb, k in zip(batch["label"], keep) if k]

        all_features.append(feat)
        all_file_paths.extend(kept_paths)
        all_slice_indices.extend(kept_slices)
        all_labels.extend(kept_labels)

    if len(all_features) == 0:
        raise RuntimeError("No features extracted. Check blank-slice filtering or input data.")

    all_features = np.concatenate(all_features, axis=0)
    return all_features, np.array(all_file_paths), np.array(all_slice_indices), np.array(all_labels)


def run_tsne(features, perplexity=30, random_state=42):
    """
    Standardize first, then t-SNE.
    """
    x = StandardScaler().fit_transform(features)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=random_state,
    )
    y = tsne.fit_transform(x)
    return y


def plot_tsne(tsne_xy, labels, out_png, title="t-SNE of slice features"):
    plt.figure(figsize=(9, 7))

    unique_labels = sorted(np.unique(labels))
    for lb in unique_labels:
        mask = labels == lb
        plt.scatter(
            tsne_xy[mask, 0],
            tsne_xy[mask, 1],
            s=18,
            alpha=0.75,
            label=lb
        )

        # label cluster center
        cx, cy = tsne_xy[mask].mean(axis=0)
        plt.text(cx, cy, lb, fontsize=10)

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=1.5, fontsize=9)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_npz(out_npz, features, file_paths, slice_indices, labels, tsne_xy):
    np.savez_compressed(
        out_npz,
        features=features,
        file_paths=file_paths,
        slice_indices=slice_indices,
        labels=labels,
        tsne=tsne_xy,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--out_npz", type=str, default="slice_features_tsne.npz")
    parser.add_argument("--out_png", type=str, default="slice_tsne.png")
    parser.add_argument("--slice_axis", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    nifti_files = list_nifti_files(args.input_dir)
    if len(nifti_files) == 0:
        raise FileNotFoundError(f"No .nii or .nii.gz found under {args.input_dir}")

    print(f"Found {len(nifti_files)} NIfTI files.")

    dataset = NiftiSliceDataset(
        nifti_files=nifti_files,
        slice_axis=args.slice_axis,
        out_size=224,
        skip_blank=True,
        blank_threshold=1e-6,
        infer_label_from_parent=True,
    )

    features, file_paths, slice_indices, labels = extract_features(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        skip_blank_runtime=True,
    )

    print("Feature matrix shape:", features.shape)

    # t-SNE can get slow for very many slices
    # For huge datasets, consider random subsampling first
    tsne_xy = run_tsne(features, perplexity=args.perplexity, random_state=42)

    save_npz(args.out_npz, features, file_paths, slice_indices, labels, tsne_xy)
    plot_tsne(tsne_xy, labels, args.out_png)

    print(f"Saved features and t-SNE to: {args.out_npz}")
    print(f"Saved plot to: {args.out_png}")


if __name__ == "__main__":
    main()