"""
Maren Salanova Florian
COCA preprocessing + dataloaders
PrediCT (segmentation task)
"""

import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

SLICE_SIZE = 512

# Basic HU windowing for CT
def apply_hu_window(vol, center=40, width=400):
    lo = center - width // 2
    hi = center + width // 2

    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo)

    return vol.astype(np.float32)


# Simple 2D augmentations
def augment_slice(img, mask):
    # random flips
    if np.random.rand() > 0.5:
        img = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()

    if np.random.rand() > 0.5:
        img = np.flipud(img).copy()
        mask = np.flipud(mask).copy()

    # rotate by 90 deg
    k = np.random.randint(0, 4)
    img = np.rot90(img, k).copy()
    mask = np.rot90(mask, k).copy()

    # small intensity noise
    noise = np.random.uniform(-0.05, 0.05)
    img = np.clip(img + noise, 0.0, 1.0)

    return img, mask


# Stratified split
def make_stratified_splits(csv_path, val_ratio=0.15, test_ratio=0.15, seed=42):
    df = pd.read_csv(csv_path)

    pos = df[df["voxels"] > 0]
    q1, q3 = pos["voxels"].quantile([0.33, 0.66])

    def get_bin(v):
        if v == 0:
            return 0
        elif v <= q1:
            return 1
        elif v <= q3:
            return 2
        else:
            return 3

    df["stratum"] = df["voxels"].apply(get_bin)

    train_val, test = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["stratum"],
        random_state=seed,
    )

    val_ratio = val_ratio / (1 - test_ratio)

    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val["stratum"],
        random_state=seed,
    )

    print(f"train: {len(train)} | val: {len(val)} | test: {len(test)}")

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# Slice-level dataset
class COCASliceDataset(Dataset):
    def __init__(
        self,
        df,
        data_dir,
        augment=False,
        hu_center=40,
        hu_width=400,
        positive_only=False,
    ):
        self.data_dir = data_dir
        self.augment = augment
        self.hu_center = hu_center
        self.hu_width = hu_width

        self.index = []
        self.volumes = {}

        for _, row in df.iterrows():
            scan_id = row["scan_id"]

            img_path = data_dir / scan_id / f"{scan_id}_img.nii.gz"
            seg_path = data_dir / scan_id / f"{scan_id}_seg.nii.gz"

            if not img_path.exists():
                continue

            img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
            seg = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path)))

            img = apply_hu_window(img, hu_center, hu_width)
            seg = seg.astype(np.float32)

            self.volumes[scan_id] = (img, seg)

            for z in range(img.shape[0]):
                has_ca = seg[z].sum() > 0

                if positive_only and not has_ca:
                    continue

                self.index.append((scan_id, z, has_ca))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        scan_id, z, _ = self.index[idx]

        img = self.volumes[scan_id][0][z].copy()
        seg = self.volumes[scan_id][1][z].copy()

        if self.augment:
            img, seg = augment_slice(img, seg)

        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        seg = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0)

        img = F.interpolate(img, size=(SLICE_SIZE, SLICE_SIZE), mode="bilinear", align_corners=False)
        seg = F.interpolate(seg, size=(SLICE_SIZE, SLICE_SIZE), mode="nearest")

        return img.squeeze(0), seg.squeeze(0), {"scan_id": scan_id, "slice": z}


# Handle class imbalance
def make_weighted_sampler(dataset):
    labels = [int(x[2]) for x in dataset.index]

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    print(f"positives: {n_pos} | negatives: {n_neg}")

    w_pos = 1.0 / n_pos if n_pos else 1.0
    w_neg = 1.0 / n_neg if n_neg else 1.0

    weights = [w_pos if l == 1 else w_neg for l in labels]

    return WeightedRandomSampler(weights, len(weights), replacement=True)


# Quick dataset stats
def print_dataset_stats(csv_path):
    df = pd.read_csv(csv_path)

    total = len(df)
    pos = (df["voxels"] > 0).sum()
    neg = total - pos

    print("\nDataset:")
    print(f"total: {total}")
    print(f"with calcium: {pos} ({pos/total*100:.1f}%)")
    print(f"no calcium: {neg} ({neg/total*100:.1f}%)")
    pos_df = df[df["voxels"] > 0]
    print(f"mean voxels (calcium scans): {pos_df['voxels'].mean():.1f}")
    print(f"median voxels (calcium scans): {pos_df['voxels'].median():.1f}")


# MAIN
if __name__ == "__main__":
    ROOT = Path(r"C:\Users\level\OneDrive\Escritorio\COCA")
    DATA_DIR = ROOT / "data_resampled"
    CSV = ROOT / "data_canonical" / "tables" / "scan_index.csv"

    print_dataset_stats(CSV)

    train_df, val_df, test_df = make_stratified_splits(CSV)

    train_ds = COCASliceDataset(train_df, DATA_DIR, augment=True)
    val_ds = COCASliceDataset(val_df, DATA_DIR)
    test_ds = COCASliceDataset(test_df, DATA_DIR)

    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    x, y, meta = next(iter(train_loader))

    print(x.shape, y.shape)
    print("range:", x.min().item(), x.max().item())
    print("mask values:", y.unique())