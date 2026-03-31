# ML4SCI GSoC 2026 — PrediCT Projects 1 & 2
**Applicant:** Maren Salanova Florian  
**Email:** marensalanova11@gmail.com  
**GitHub:** [github.com/MakiBuu](https://github.com/MakiBuu)  
**University:** Universidad Pública de Navarra, Spain  
**Organization:** Machine Learning for Science (ML4SCI)  

## Common Task — COCA Dataset Preprocessing

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Total scans | 787 |
| Scans with calcium | 447 (56.8%) |
| Scans without calcium | 340 (43.2%) |
| Mean calcium voxels (positive scans) | 1034.4 |
| Median calcium voxels (positive scans) | 334.0 |
| Train / Val / Test split | 550 / 118 / 119 |
| Positive slices (calcium present) | 2550 (9.4%) |
| Negative slices | 24704 (90.6%) |

### Justification

HU windowing was applied using center=40 and width=400 (range −160 to 240 HU), which is the standard cardiac soft-tissue window for CT. This choice suppresses irrelevant extremes such as air and dense bone while preserving the contrast of soft cardiac structures and calcium deposits, which appear at the upper end of this range. Data augmentation consists of random horizontal and vertical flips, 90-degree rotations, and small intensity jitter (±5%), applied identically to both image and mask tensors to strictly preserve spatial correspondence — a non-negotiable requirement for segmentation tasks where any misalignment between input and label would corrupt training. The stratified train/val/test split uses calcium burden quartiles (no calcium, low, medium, high) as strata, ensuring that each split contains a representative distribution of disease severity rather than an accidental concentration of easy negative cases in training. Finally, a WeightedRandomSampler addresses the severe class imbalance at the slice level: calcium-positive slices represent only 9.4% of all slices, meaning a naive training loop would learn to predict all-zero masks and achieve artificially high accuracy. The sampler upweights positive slices so each training batch contains a balanced mix, forcing the model to learn meaningful calcium boundaries.

---

## Project 1 — Heart Segmentation Model

### Results

| Metric | Value |
|--------|-------|
| Mean Dice (test set) | **0.8515** |
| Median Dice (test set) | 0.8851 |
| Scans with Dice > 0.85 | 5 / 8 |
| Mean U-Net inference time | 1.1s / scan |
| Mean TotalSegmentator time | 31.1s / scan |
| Speedup over TotalSegmentator | **29.2x** |
| Model parameters | ~820K |

### Model Justification

A lightweight 2D U-Net implemented in MONAI was selected for three reasons. First, the U-Net architecture with skip connections is the established baseline for medical image segmentation and has been validated extensively on cardiac CT, making it a well-understood and reproducible choice for this task. Second, operating slice-by-slice in 2D rather than volumetrically in 3D drastically reduces memory requirements and inference time — enabling the 29x speedup over TotalSegmentator observed on CPU, which directly addresses the project goal of producing a faster alternative for heart localization. Third, the relatively small channel configuration (32, 64, 128, 256) keeps the model at ~820K parameters, sufficient to learn the coarse heart boundary from 32 training scans while avoiding overfitting on a limited ground truth set. The model achieved a mean Dice of 0.8515 against TotalSegmentator ground truth, meeting the target, with a median of 0.8851 indicating robust performance across the majority of test cases.

---

## Project 2 — Radiomics Feature Extraction and Phenotyping

### Agatston Score Distribution (30 scans)

| Category | Count |
|----------|-------|
| 0 (no calcium) | 10 |
| 1-99 (mild) | 9 |
| 100-399 (moderate) | 4 |
| >=400 (severe) | 7 |

### Significant Associations — Spearman (p < 0.05)

| Feature | Spearman rho | p-value |
|---------|-------------|---------|
| calcium_volume_mm3 | 0.9955 | <0.001 |
| hu_max | 0.9805 | <0.001 |
| shape_Maximum3DDiameter | 0.9516 | <0.001 |
| glcm_Contrast | 0.9378 | <0.001 |
| hu_mean | 0.9260 | <0.001 |
| glszm_LargeAreaEmphasis | 0.7911 | <0.001 |
| glrlm_ShortRunEmphasis | 0.7703 | <0.001 |
| glrlm_RunPercentage | 0.7583 | <0.001 |
| glcm_Correlation | 0.4137 | 0.023 |

All 13 features passed Kruskal-Wallis significance (p < 0.001) across Agatston categories.

### Justification

PyRadiomics was selected for feature extraction because it is the standard open-source library for reproducible radiomics in medical imaging, with well-validated implementations of shape, texture, and run-length features. Features were extracted directly from the calcium segmentation masks in the resampled COCA scans, covering four feature classes: Shape (morphological descriptors of calcium deposits), GLCM (grey-level co-occurrence matrix texture), GLSZM (grey-level size zone matrix), and GLRLM (grey-level run-length matrix). Agatston scores were computed at original voxel spacing to comply with the clinical standard, then used as the continuous outcome for Spearman correlation and the categorical grouping variable for Kruskal-Wallis testing. All 14 extracted features showed statistically significant associations with Agatston score (p < 0.05), with calcium volume (rho=0.9955) and maximum HU (rho=0.9805) showing near-perfect correlation — consistent with the clinical definition of Agatston scoring. t-SNE visualization and KMeans clustering (k=4) further confirm that radiomics features naturally separate scans into groups corresponding to Agatston severity categories.

---

## How to Reproduce

### Requirements
```bash
pip install SimpleITK pandas numpy torch monai scikit-learn tqdm matplotlib \
            jupyter TotalSegmentator pyradiomics seaborn scipy umap-learn
```

### Steps
```bash
# 1. Process raw DICOM data
python common_task/COCA_processor.py

# 2. Resample to uniform spacing
python common_task/COCAResampler.py

# 3. Build preprocessing pipeline and splits
python common_task/pipeline.py

# --- Project 1 ---
# 4. Generate heart ground truth (requires TotalSegmentator academic license)
python project1_segmentation/generate_labels.py

# 5. Train U-Net
python project1_segmentation/train.py

# 6. Evaluate segmentation
jupyter notebook project1_segmentation/evaluate.ipynb

# --- Project 2 ---
# 7. Extract radiomics features + Agatston scores
python project2_radiomics/extract_features.py

# 8. Statistical analysis and visualizations
jupyter notebook project2_radiomics/analysis.ipynb
```
