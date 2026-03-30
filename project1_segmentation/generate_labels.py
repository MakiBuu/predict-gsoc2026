"""
generate_labels.py
Runs TotalSegmentator on 40 COCA scans to produce heart segmentation ground truth.
"""

import subprocess
import time
import json
import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path


# ── Config

ROOT         = Path(r"C:\Users\level\OneDrive\Escritorio\COCA")
RESAMPLED    = ROOT / "data_resampled"
CSV_PATH     = ROOT / "data_canonical" / "tables" / "scan_index.csv"
LABELS_DIR   = ROOT / "heart_labels"        # TotalSegmentator output goes here
GT_DIR       = ROOT / "heart_gt"            # final single-file masks go here
N_SCANS      = 40                           # how many scans to annotate
SEED         = 42

LABELS_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR.mkdir(parents=True, exist_ok=True)


# ── Sample scans 

df = pd.read_csv(CSV_PATH)

pos = df[df["voxels"] > 0].sample(n=25, random_state=SEED)
neg = df[df["voxels"] == 0].sample(n=15, random_state=SEED)
selected = pd.concat([pos, neg]).reset_index(drop=True)
selected.to_csv(ROOT / "data_canonical" / "tables" / "label_scans.csv", index=False)

print(f"Selected {len(selected)} scans ({len(pos)} with calcium, {len(neg)} without)")


# ── Run TotalSegmentator

timings = {}
failed  = []

for i, row in selected.iterrows():
    scan_id  = row["scan_id"]
    img_path = RESAMPLED / scan_id / f"{scan_id}_img.nii.gz"
    out_dir  = LABELS_DIR / scan_id

    if not img_path.exists():
        print(f"[SKIP] {scan_id} — image not found")
        continue

    # Skip if already done
    if (out_dir / "heart.nii.gz").exists():
        print(f"[SKIP] {scan_id} — already processed")
        continue

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{i+1}/{len(selected)}] Running TotalSegmentator on {scan_id}...")

    t0 = time.time()
    result = subprocess.run(
        [
            "TotalSegmentator",
            "-i", str(img_path),
            "-o", str(out_dir),
            "--fast",                   # faster inference, good enough for heart
            "--roi_subset", "heart",    # only segment the heart
        ],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    timings[scan_id] = round(elapsed, 2)

    if result.returncode != 0:
        print(f"  [ERROR] {scan_id}:\n{result.stderr[-300:]}")
        failed.append(scan_id)
    else:
        print(f"  Done in {elapsed:.1f}s")


print("\nConsolidating masks...")
processed = []

for _, row in selected.iterrows():
    scan_id   = row["scan_id"]
    heart_src = LABELS_DIR / scan_id / "heart.nii.gz"
    heart_dst = GT_DIR / f"{scan_id}_heart.nii.gz"

    if not heart_src.exists():
        continue

    import shutil
    shutil.copy2(heart_src, heart_dst)

    # Quick sanity check: count non-zero voxels
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(heart_dst)))
    voxels = int(np.sum(mask > 0))
    processed.append({"scan_id": scan_id, "heart_voxels": voxels,
                       "ts_time_s": timings.get(scan_id, None)})
    print(f"  {scan_id}: {voxels} heart voxels")

results_df = pd.DataFrame(processed)
results_df.to_csv(ROOT / "data_canonical" / "tables" / "label_results.csv", index=False)

print(f"\nDone. {len(processed)} masks saved to {GT_DIR}")
print(f"Failed: {failed}")
print(f"Average TotalSegmentator time: {np.mean([t for t in timings.values()]):.1f}s per scan")