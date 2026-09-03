# Kidney CT Multi-Class Classification (PyTorch)

A deep learning pipeline for classifying kidney CT scan slices into four categories — **Cyst, Normal, Stone, Tumor** — using transfer learning, with an emphasis on rigorous evaluation and interpretability rather than raw accuracy alone.

> **Status:** Research prototype / academic project. **Not a diagnostic tool.** Not clinically validated. See [Limitations](#limitations--ethical-considerations) before drawing any conclusions from this work.

---

## Overview

This project explores automated classification of kidney pathology from CT imaging as a computer vision problem, with particular attention to failure modes that are common — and often under-reported — in medical imaging ML projects: data leakage, class imbalance, and models learning spurious shortcuts instead of anatomically meaningful features.

**Key design decisions:**
- Transfer learning (ResNet50, ImageNet-pretrained) rather than training from scratch, given limited data
- Two-phase training: frozen backbone → partial fine-tuning
- Per-class evaluation (precision/recall/F1/AUC), not just overall accuracy — false negatives on Tumor are far more consequential than on Cyst
- Grad-CAM visualization to check *why* the model predicts what it predicts, not just *whether* it's right
- Explicit documentation of what wasn't verified (e.g. patient-level leakage), rather than silently reporting an inflated number

---

## Dataset

- 4 classes: `Cyst`, `Normal`, `Stone`, `Tumor`
- Axial and coronal CT slices, 512×512 JPG, resized to 224×224 for the model
- Pre-split into `train / val / test` folders, each containing the four class subfolders
- Class counts: *[fill in — run the class-count cell and paste the numbers here]*

**Known dataset caveat:** slices appear to be pre-split without confirmed patient-level separation. If the same patient's slices span multiple splits, reported metrics may be optimistic. This is flagged explicitly rather than hidden — see [Limitations](#limitations--ethical-considerations).

---

## Method

### Preprocessing
- Resize to 224×224
- Grayscale → 3-channel (to match ImageNet-pretrained weight expectations)
- Normalization using ImageNet mean/std
- Augmentation (train only): mild rotation (±10°), random resized crop, brightness/contrast jitter
- No horizontal flip (not confirmed anatomically safe given mixed axial/coronal views in the data)

### Model
- **Backbone:** ResNet50 (ImageNet-pretrained)
- **Phase 1:** backbone frozen, new classification head trained (Adam, lr=1e-3)
- **Phase 2:** `layer4` + head unfrozen, fine-tuned at lower LR (Adam, lr=1e-5)
- **Loss:** weighted cross-entropy (weights computed from train-split class frequencies to address imbalance)
- **Comparison model:** EfficientNet-B0 (ImageNet-pretrained) — *[state whether you ran this]*

### Evaluation
- Macro accuracy, per-class precision/recall/F1
- Macro ROC-AUC (one-vs-rest)
- Confusion matrix
- Grad-CAM on `layer4[-1]` for qualitative inspection of model attention

---

## Results

*[Fill in after training — do not leave placeholder numbers in a public repo]*

| Class  | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| Cyst   |           |        |          |
| Normal |           |        |          |
| Stone  |           |        |          |
| Tumor  |           |        |          |

- **Overall macro accuracy:** —
- **Macro ROC-AUC:** —
- **Confusion matrix:** see `outputs/confusion_matrix.png` *(or embed the image here)*

### Grad-CAM findings
*[Briefly describe what you observed — e.g. "Model attention consistently localized to the kidney region across all four classes" or "Attention occasionally fell on scan borders for the Stone class, suggesting possible shortcut learning — flagged for further investigation."]*

This qualitative check matters more than it might seem: a model attending to the correct anatomy is meaningfully more trustworthy than one with a high score and no attention-map inspection.

---

## Repository Structure

```
├── kidney_classification_pytorch.ipynb   # Full training + evaluation notebook (Colab-ready)
├── README.md
└── outputs/                              # (optional) saved plots, confusion matrix, sample Grad-CAMs
```

---

## How to Run

1. Upload the dataset to Google Drive, preserving the `train/val/test → Cyst/Normal/Stone/Tumor` structure.
2. Open `kidney_classification_pytorch.ipynb` in Google Colab.
3. Set runtime to GPU (Runtime → Change runtime type → T4 GPU).
4. Update `DATA_ROOT` in the config cell to your Drive path.
5. Run all cells in order.

Trained model weights are saved to Drive as `resnet50_kidney_classifier.pt`.

---

## Limitations & Ethical Considerations

- **Not a diagnostic tool.** This is a research/educational computer vision system. It has not been evaluated against clinical ground truth by qualified radiologists, nor validated on an external dataset.
- **Possible data leakage.** If dataset slices are patient-derived and patient IDs were not used to enforce the train/val/test split, the same patient's anatomy could appear across splits, inflating reported performance. This has not been independently confirmed for this dataset.
- **No external validation.** Results reflect this dataset's distribution only (single source, unknown scanner/protocol variation) and may not generalize to CT scans from other institutions or populations.
- **Slice-level, not volumetric.** The model classifies individual 2D slices; it does not use full 3D CT context, which a clinical system realistically would need.
- **Class imbalance mitigation (class weighting) reduces but does not eliminate** the risk of biased performance across classes — per-class metrics should always be read alongside the overall accuracy, not instead of it.
- **Grad-CAM is a sanity check, not proof of correctness.** Reasonable-looking attention maps do not guarantee the model has learned clinically meaningful or generalizable features.

Any future extension of this work toward real-world use would require: clinical-grade labeled data with verified patient-level provenance, external validation on independent datasets, radiologist-reviewed ground truth, and prospective evaluation — none of which are in scope here.

---

## Future Work

- Patient-level split verification (or re-split if metadata is available)
- K-fold cross-validation with mean ± std reporting across seeds
- Full EfficientNet-B0 / DenseNet121 comparison table
- 3D/volumetric modeling using full CT series rather than isolated slices
- External dataset validation

---

## Author

*[Your name]* — *[program/university]*
Built as an independent project exploring applied deep learning for medical image classification.
