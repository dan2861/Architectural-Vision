# Architectural-Vision
## 1  Project Overview
- **Objective** Classify photographs of Yale buildings into **Gothic, Brutalist, or Colonial** styles.  
- **Method** Use *frozen* convolutional networks as universal feature extractors, then train lightweight, interpretable classifiers on those features.  
- **Success target** ≥ 60 % accuracy on a held‑out Yale‑photo test set, with confusion‑matrix insights into typical style confusions.

---

## 2  Data Pipeline
### 2.1 Sources  
1. **Kaggle Architecture Styles** – filtered to the three target classes.  
2. **Your own Yale photos** – at least 30 – 40 per class, taken under varied lighting.

### 2.2 Curation & Storage  
```
data/
  train/
    gothic/…
    brutalist/…
    colonial/…
  yale_test/
    gothic/…
    brutalist/…
    colonial/…
```

### 2.3 Augmentation Strategy  
- **Always**: horizontal flip, ±10° rotation, random 0.8 – 1.2 scaling.  
- **When domain mismatch shows**: CLAHE or histogram equalisation.  
> Keep augmentation identical for all later experiments so results are comparable.

---

## 3  Feature‑Extraction Architecture
```
image (RGB, 224×224)
   │
   ▼
┌─────────────┐   ResNet‑50 / VGG16 / MobileNetV2  (weights frozen)
│ conv layers │──► feature map  (C×H×W)
└─────────────┘
   │
   ├─ Option A: Global Average Pooling   →  C‑dim vector
   ├─ Option B: Max Pooling              →  C‑dim vector
   ├─ Option C: Covariance (Gram) Pool   →  C(C+1)/2  upper‑tri
   └─ Option D: Bag‑of‑Deep‑Features     →  k‑dim histogram
```

**Dimensionality control** (choose at run‑time):  
- **None** keep raw vector.  
- **PCA–95 % var** retain as many components as needed for 95 % variance.  
- **Fixed 128‑D** for extreme compactness / k‑NN speed.

All variants are written to `.npz` with identical ordering so they can be concatenated or swapped in later.

---

## 4  Model Layer
For every feature variant:

| Model           | Hyper‑params            | Notes |
|-----------------|-------------------------|-------|
| Linear SVM      | C ∈ {0.1, 1, 10}       | Fast baseline |
| RBF‑SVM         | C, γ log‑grid          | Use after PCA; γ sensitive to dim |
| Logistic Reg.   | C ∈ {0.1, 1, 10}       | Gives calibrated probs |
| k‑NN            | k ∈ {1, 3, 5}          | Cosine distance after ℓ2‑norm |
| MLP (1 layer)   | hidden {64,128}, α=1e‑4| Keep tiny to avoid over‑fit |

All hyper‑parameter grids are searched with **stratified 5‑fold CV on Kaggle** only. Final model is frozen, then evaluated once on the Yale test split.

---

## 5  Evaluation Suite
```
metrics/
  accuracy.txt
  classification_report.json
  confusion_matrix.png
  tsne_2d.png
```
- **Primary** Accuracy.  
- **Insight** Confusion matrix; macro F1.  
- **Visual** t‑SNE of each best‑performing feature space, colour‑coded by label, to sanity‑check separability.

---

## 6  Implementation Road‑Map (single‑threaded but modular)
1. **Dataset builder (`build_dataset.py`)**  
   - Download / copy sources, enforce directory schema, write `metadata.csv`.

2. **Augmentation runner (`augment.py`)**  
   - Save augmented images beside originals; use deterministic seeds.

3. **Feature extractor (`deep_features.py`)**  
   - CLI flags: `--arch`, `--pool`, `--pca` (`none|95var|128`), `--split train/yale_test`.  
   - Outputs: `features/{split}_{arch}_{pool}_{pca}.npz`.

4. **Model trainer (`train.py`)**  
   - Loads `.npz`; runs grid search; saves `model.pkl` + CV scores.

5. **Evaluator (`evaluate.py`)**  
   - Loads frozen model; reports metrics; writes plots.

6. **Experiment driver (`run_experiments.sh`)**  
   - Bash or Makefile enumerating feature‑arch‑pool combinations:
     ```
     for arch in resnet50 vgg16 mobilenetv2; do
       for pool in gap gmp gram bodf; do
         for pca in none 95var 128; do
             python deep_features.py --arch $arch --pool $pool --pca $pca
             python train.py --features $fname
         done
       done
     done
     ```

7. **Result aggregator (`summarise.py`)**  
   - Combines all `accuracy.txt` into a CSV leaderboard, highlights top‑k.

---

## 7  Decision Points & Scenarios
| If you… | …then do this | Rationale |
|---------|---------------|-----------|
| **Need faster training** | Use MobileNetV2 + GAP + PCA‑128 | 128‑D vectors train SVMs in milliseconds |
| **Care about texture detail** | Try Gram pooling before PCA | Captures pairwise filter co‑activation useful for masonry vs. concrete |
| **See over‑fit** | Increase PCA compression **and** use L2‑regularised Logistic Regression | Reduces variance and keeps interpretability |
| **Accuracy plateaus < 60 %** | Fuse *two* architectures (concat ResNet‑GAP + VGG‑Gram) before PCA | Heterogeneous features often boost discriminative power |
| **Need interpretability for report** | Stick to ResNet‑GAP, Linear SVM, and use **grad‑CAM** on the conv maps | Provides heatmaps of building parts influencing decisions |

---

## 8  Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Legal limits on Google Street View | Training data unusable | Use only self‑captured or CC‑licensed images |
| Domain shift (ImageNet → Yale photos) | Poor generalisation | Histogram equalisation; add colour‑jitter during augmentation |
| Small Yale test set | High variance in metrics |  Bootstrap confidence intervals or collect more photos |

---

## 9  Timeline (4‑week sprint)
| Week | Deliverable |
|------|-------------|
| 1 | Dataset ready; baseline ResNet‑GAP features extracted |
      All pooling variants implemented; first accuracy numbers |
| 2 | Full hyper‑param sweep; best model frozen; failure analysis |
      Write‑up, plots, code refactor, “top‑k mistakes” appendix |

---

### Next Action
1. Clone repo skeleton: `git clone ... && cd yale‑style‑classifier`.  
2. Start **Task 1** (`build_dataset.py`) today; once raw images are in place, Tasks 2‑3 can run in parallel on a laptop GPU or Colab.

Feel free to adapt filenames or swap any library (e.g., TensorFlow instead of PyTorch) as long as the artefact contract—images → `.npz` → `model.pkl`—stays fixed.