"""
train.py
Trains a lightweight 2D U-Net (MONAI) to predict heart segmentation masks.
"""

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete
from monai.data import decollate_batch


# Config 

ROOT       = Path(r"C:\Users\level\OneDrive\Escritorio\COCA")
RESAMPLED  = ROOT / "data_resampled"
GT_DIR     = ROOT / "heart_gt"
LABEL_CSV  = ROOT / "data_canonical" / "tables" / "label_results.csv"
MODEL_DIR  = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SLICE_SIZE = 256       # smaller than pipeline.py to keep training fast
BATCH_SIZE = 8
EPOCHS     = 60
LR         = 1e-3
VAL_RATIO  = 0.2
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")


# HU windowing

def apply_hu_window(vol, center=40, width=400):
    lo, hi = center - width // 2, center + width // 2
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / (hi - lo)).astype(np.float32)


# Dataset 

class HeartSliceDataset(Dataset):
    def __init__(self, scan_ids, resampled_dir, gt_dir, augment=False):
        self.augment = augment
        self.index   = []   # (img_slice, seg_slice)

        for scan_id in scan_ids:
            img_path  = resampled_dir / scan_id / f"{scan_id}_img.nii.gz"
            mask_path = gt_dir / f"{scan_id}_heart.nii.gz"

            if not img_path.exists() or not mask_path.exists():
                continue

            img  = apply_hu_window(sitk.GetArrayFromImage(sitk.ReadImage(str(img_path))))
            mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(np.float32)

            # Binarise (TotalSegmentator uses label 1 for heart)
            mask = (mask > 0).astype(np.float32)

            for z in range(img.shape[0]):
                self.index.append((img[z].copy(), mask[z].copy()))

    def __len__(self):
        return len(self.index)

    def _resize(self, arr, mode):
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        t = F.interpolate(t, size=(SLICE_SIZE, SLICE_SIZE), mode=mode,
                          align_corners=False if mode == "bilinear" else None)
        return t.squeeze(0)   # (1,H,W)

    def __getitem__(self, idx):
        img, mask = self.index[idx]

        if self.augment:
            # Horizontal flip
            if np.random.rand() > 0.5:
                img  = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            # Vertical flip
            if np.random.rand() > 0.5:
                img  = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
            # 90-degree rotation
            k = np.random.randint(0, 4)
            img  = np.rot90(img,  k).copy()
            mask = np.rot90(mask, k).copy()
            # Intensity jitter (image only)
            img = np.clip(img + np.random.uniform(-0.05, 0.05), 0.0, 1.0)

        img_t  = self._resize(img,  "bilinear")
        mask_t = self._resize(mask, "nearest")

        return img_t, mask_t


#Build splits

label_df  = pd.read_csv(LABEL_CSV)
scan_ids  = label_df["scan_id"].tolist()

np.random.seed(42)
np.random.shuffle(scan_ids)
n_val    = int(len(scan_ids) * VAL_RATIO)
val_ids  = scan_ids[:n_val]
train_ids = scan_ids[n_val:]

print(f"Train scans: {len(train_ids)} | Val scans: {len(val_ids)}")

train_ds = HeartSliceDataset(train_ids, RESAMPLED, GT_DIR, augment=True)
val_ds   = HeartSliceDataset(val_ids,   RESAMPLED, GT_DIR, augment=False)

print(f"Train slices: {len(train_ds)} | Val slices: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


#Model

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,                         # background + heart
    channels=(32, 64, 128, 256),             # small — fast on CPU too
    strides=(2, 2, 2),
    num_res_units=1,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_fn    = DiceLoss(to_onehot_y=True, softmax=True)
dice_metric = DiceMetric(include_background=False, reduction="mean")

post_pred  = lambda x: AsDiscrete(argmax=True, to_onehot=2)(Activations(softmax=True)(x))
post_label = AsDiscrete(to_onehot=2)


# Training loop

best_val_dice = 0.0
history = {"train_loss": [], "val_dice": []}

for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()
    epoch_loss = 0.0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]", leave=False):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = loss_fn(preds, masks.long())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    history["train_loss"].append(avg_loss)

    model.eval()
    dice_metric.reset()
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            preds_list  = decollate_batch(preds)
            masks_list  = decollate_batch(masks.long())
            preds_post  = [post_pred(p)  for p in preds_list]
            masks_post  = [post_label(m) for m in masks_list]
            dice_metric(y_pred=preds_post, y=masks_post)

    val_dice = dice_metric.aggregate().item()
    history["val_dice"].append(val_dice)
    scheduler.step()

    print(f"Epoch {epoch:3d} | loss: {avg_loss:.4f} | val Dice: {val_dice:.4f}")

    # Save best
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), MODEL_DIR / "best_unet.pth")
        print(f"  ✓ New best saved ({val_dice:.4f})")

print(f"\nTraining complete. Best val Dice: {best_val_dice:.4f}")
pd.DataFrame(history).to_csv(MODEL_DIR / "training_history.csv", index=False)