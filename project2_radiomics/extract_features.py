"""
Maren Salanova Florian
extract_features.py
Project 2 - Radiomics & Phenotyping
Extracts PyRadiomics features + Agatston scores from 20-30 COCA scans.
"""

import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import radiomics
from radiomics import featureextractor
import logging

# Suppress verbose PyRadiomics logs
logging.getLogger("radiomics").setLevel(logging.ERROR)

# Config

ROOT        = Path(r"C:\Users\level\OneDrive\Escritorio\COCA")
RESAMPLED   = ROOT / "data_resampled"
CSV_PATH    = ROOT / "data_canonical" / "tables" / "scan_index.csv"
OUT_DIR     = ROOT / "project2_radiomics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SCANS     = 30
SEED        = 42

# PyRadiomics settings — extract exactly the features required
PARAMS = {
    "imageType": {
        "Original": {}
    },
    "featureClass": {
        "shape": [
            "Sphericity",
            "SurfaceVolumeRatio",
            "Maximum3DDiameter",
        ],
        "glcm": [
            "Contrast",
            "Correlation",
            "Id",
        ],
        "glszm": [
            "SmallAreaEmphasis",
            "LargeAreaEmphasis",
            "ZonePercentage",
        ],
        "glrlm": [
            "ShortRunEmphasis",
            "LongRunEmphasis",
            "RunPercentage",
        ],
    },
    "setting": {
        "binWidth": 25,
        "resampledPixelSpacing": None,
        "interpolator": "sitkBSpline",
     }   
}

extractor = featureextractor.RadiomicsFeatureExtractor(PARAMS)


# Agatston Score

def compute_agatston(img_sitk: sitk.Image, mask_sitk: sitk.Image) -> float:
    """
    Computes Agatston score using original image spacing (not resampled).
    Formula: sum over slices of (area_mm2 * density_factor)
    Density factors: 130-199 HU → 1, 200-299 → 2, 300-399 → 3, ≥400 → 4
    Only voxels with HU ≥ 130 inside the mask are counted.
    """
    img_arr  = sitk.GetArrayFromImage(img_sitk).astype(np.float32) 
    mask_arr = sitk.GetArrayFromImage(mask_sitk).astype(np.uint8)

    spacing  = img_sitk.GetSpacing()   
    pixel_area_mm2 = spacing[0] * spacing[1]

    def density_factor(hu):
        if hu >= 400: return 4
        if hu >= 300: return 3
        if hu >= 200: return 2
        if hu >= 130: return 1
        return 0

    agatston = 0.0
    for z in range(img_arr.shape[0]):
        slice_img  = img_arr[z]
        slice_mask = mask_arr[z]

        # Only consider masked voxels with HU >= 130
        calcium_voxels = slice_img[(slice_mask > 0) & (slice_img >= 130)]
        if len(calcium_voxels) == 0:
            continue

        max_hu   = float(calcium_voxels.max())
        df       = density_factor(max_hu)
        area_mm2 = len(calcium_voxels) * pixel_area_mm2
        agatston += area_mm2 * df

    return round(agatston, 2)


def agatston_category(score: float) -> str:
    if score == 0:       return "0"
    if score < 100:      return "1-99"
    if score < 400:      return "100-399"
    return ">=400"


#HU statistics

def calcium_hu_stats(img_sitk: sitk.Image, mask_sitk: sitk.Image) -> dict:
    img_arr  = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(mask_sitk).astype(np.uint8)

    calcium = img_arr[(mask_arr > 0) & (img_arr >= 130)]
    if len(calcium) == 0:
        return {"hu_max": 0.0, "hu_mean": 0.0, "calcium_volume_mm3": 0.0}

    spacing = img_sitk.GetSpacing()
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]

    return {
        "hu_max":              round(float(calcium.max()), 2),
        "hu_mean":             round(float(calcium.mean()), 2),
        "calcium_volume_mm3":  round(len(calcium) * voxel_vol_mm3, 2),
    }


# Sample scans

df = pd.read_csv(CSV_PATH)

# Stratified: mix calcium / no-calcium
pos = df[df["voxels"] > 0].sample(n=20, random_state=SEED)
neg = df[df["voxels"] == 0].sample(n=10, random_state=SEED)
selected = pd.concat([pos, neg]).reset_index(drop=True)

print(f"Extracting features from {len(selected)} scans "
      f"({len(pos)} with calcium, {len(neg)} without)...")


# Main extraction loop

rows = []
failed = []

for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Extracting"):
    scan_id  = row["scan_id"]
    img_path = RESAMPLED / scan_id / f"{scan_id}_img.nii.gz"
    seg_path = RESAMPLED / scan_id / f"{scan_id}_seg.nii.gz"

    if not img_path.exists() or not seg_path.exists():
        failed.append(scan_id)
        continue

    try:
        img_sitk  = sitk.ReadImage(str(img_path))
        mask_sitk = sitk.ReadImage(str(seg_path))

        agatston = compute_agatston(img_sitk, mask_sitk)
        cat      = agatston_category(agatston)
        hu_stats = calcium_hu_stats(img_sitk, mask_sitk)

        record = {
            "scan_id":          scan_id,
            "patient_id":       row["patient_id"],
            "calcium_voxels":   row["voxels"],
            "agatston_score":   agatston,
            "agatston_category": cat,
            **hu_stats,
        }

        # PyRadiomics (only extract if mask has calcium) 
        if int(mask_sitk.GetSize()[0]) > 0 and row["voxels"] > 0:
            result = extractor.execute(img_sitk, mask_sitk, label=1)
            for key, val in result.items():
                if key.startswith("original_"):
                    short_key = key.replace("original_", "")
                    try:
                        record[short_key] = float(val)
                    except (TypeError, ValueError):
                        record[short_key] = None
        else:
            # Zero-calcium scan: fill radiomics features with 0
            for fc in ["shape", "glcm", "glszm", "glrlm"]:
                for feat in PARAMS["featureClass"][fc]:
                    record[f"{fc}_{feat}"] = 0.0

        rows.append(record)

    except Exception as e:
        print(f"\n  [ERROR] {scan_id}: {e}")
        failed.append(scan_id)

# Save CSV

features_df = pd.DataFrame(rows)
out_csv = OUT_DIR / "radiomics_features.csv"
features_df.to_csv(out_csv, index=False)

print(f"\nDone. {len(features_df)} scans processed. Failed: {len(failed)}")
print(f"Feature CSV saved to: {out_csv}")
print(f"Columns: {list(features_df.columns)}")

# Quick Agatston distribution
print("\nAgatston category distribution:")
print(features_df["agatston_category"].value_counts().to_string())