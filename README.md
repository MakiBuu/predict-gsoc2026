# ML4SCI GSoC 2026 — PrediCT Project 1: Heart Segmentation
**Applicant:** Maren Salanova Florian  
**Project:** PrediCT Project 1 — Calcium/Heart Segmentation  
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

HU windowing was applied using center=40 and width=400 (range −160 to 240 HU), which is the standard cardiac soft-tissue window for CT. This choice suppresses irrelevant extremes such as air and dense bone while preserving the contrast of soft cardiac structures and calcium deposits, which appear at the upper end of this range. Data augmentation consists of random horizontal and vertical flips, 90-degree rotations, and small intensity jitter (±5%), applied identically to both image and mask tensors to strictly preserve spatial correspondence — a non-negotiable requirement for segmentation tasks where any misalignment between input and label would corrupt training. The stratified train/val/test split uses calcium burden quartiles (no calcium, low, medium, high) as strata, ensuring that each split contains a representative distribution of disease severity rather than an accidental concentration of easy negative cases in training. Finally, a `WeightedRandomSampler` addresses the severe class imbalance at the slice level: calcium-positive slices represent only 9.4% of all slices, meaning a naive training loop would learn to predict all-zero masks and achieve artificially high accuracy. The sampler upweights positive slices so each training batch contains a balanced mix, forcing the model to learn meaningful calcium boundaries.

---

## Project 1 — Heart Segmentation Model

### Results

| Metric | Value |
|--------|-------|
| Mean Dice (test set) | **0.8515** |
| Median Dice (test set) | 0.8851 |
| Scans with Dice > 0.85 | 5 / 8 |
| Mean U-Net inference time | 1.1s per scan |
| Mean TotalSegmentator time | 31.1s per scan |
| Speedup | **29.2× faster** |

### Model Justification

A lightweight 2D U-Net implemented in MONAI was selected for three reasons. First, the U-Net architecture with skip connections is the established baseline for medical image segmentation and has been validated extensively on cardiac CT, making it a well-understood and reproducible choice for this task. Second, operating slice-by-slice in 2D rather than volumetrically in 3D drastically reduces memory requirements and inference time — enabling the 29× speedup over TotalSegmentator observed on CPU, which directly addresses the project goal of producing a faster alternative to TotalSegmentator for heart localization. Third, the relatively small channel configuration (32, 64, 128, 256) keeps the model at ~820K parameters, which is sufficient to learn the coarse heart boundary from 32 training scans while avoiding overfitting on a limited ground truth set. The model achieved a mean Dice of 0.8515 against TotalSegmentator ground truth, meeting the >0.85 target, with a median of 0.8851 indicating robust performance across the majority of test cases.

---

## How to Reproduce

### Requirements
```bash
pip install SimpleITK pandas numpy torch monai scikit-learn tqdm matplotlib jupyter TotalSegmentator
```

### Steps
```bash
# 1. Process raw DICOM data
python common_task/COCA_processor.py

# 2. Resample to uniform spacing
python common_task/COCAResampler.py

# 3. Build preprocessing pipeline and splits
python common_task/pipeline.py

# 4. Generate heart ground truth with TotalSegmentator (requires academic license)
python project1_segmentation/generate_labels.py

# 5. Train U-Net
python project1_segmentation/train.py

# 6. Evaluate
jupyter notebook project1_segmentation/evaluate.ipynb
```
