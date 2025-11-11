# 🏛️ Architectural-Vision: Classifying Yale Architecture Using Machine Learning

> A machine learning project that classifies Yale campus buildings into **Gothic**, **Brutalist**, or **Colonial** architectural styles using deep feature extraction from pretrained CNNs and lightweight classifiers.

---

## 📘 Overview

Yale University’s campus features a rich blend of architectural styles that define its cultural and historical character. This project explores how **machine learning**—specifically **transfer learning** and **classical classification techniques**—can automatically distinguish between these architectural types based on images.

By leveraging a pretrained **ResNet-50** CNN as a frozen feature extractor and training a **Support Vector Machine (SVM)** classifier on deep feature embeddings, this project demonstrates a hybrid approach combining **deep learning feature richness** with **classical model interpretability**.

---

## 👥 Contributors

| Name | Role |
|------|------|
| Daniel Metaferia | Lead Developer, ML Engineer |
| Nour Darragi | Data Engineer, Preprocessing |
| Brian Di Bassinga | Model Evaluation, Visualization |
| Mike Masamvu | Data Collection, Augmentation |

---

## 📂 Project Structure

Each step of the workflow is modularized in separate notebooks for clarity and reproducibility:

| File | Description |
|------|--------------|
| `01_build_dataset.ipynb` | Mounts dataset, organizes files, and builds `metadata.csv` mapping image paths to labels. |
| `02_augment.ipynb` | Performs deterministic augmentations (flip, rotate, scale) to improve generalization. |
| `03_deep_features.ipynb` | Extracts deep features using frozen ResNet-50 and applies PCA for dimensionality reduction. |
| `04_train_models.ipynb` | Trains and evaluates multiple classifiers: CNN, Logistic Regression, Perceptron, and SVM. |
| `05_visualize_results.ipynb` *(optional)* | Generates confusion matrices, t-SNE visualizations, and accuracy reports. |

📎 **Dataset Folder Structure**
```
data/
│
├── kaggle_dataset/          # Source training data
├── yale_images/             # Collected test images
├── augmented/               # Generated augmentations
└── metadata.csv             # Image path-to-label mapping
```

---

## 🧠 Methodology

### 1. Problem Definition  
A supervised image classification task:  
**Input:** Image of a Yale building  
**Output:** Predicted architectural style – *Gothic*, *Brutalist*, or *Colonial*  

### 2. Data Collection & Preprocessing  
- **Training Data:** Kaggle Architecture Styles dataset ([link](https://www.kaggle.com/datasets/wwymak/architecture-dataset))  
- **Testing Data:** Hand-collected Yale campus images (resized to 224×224)  
- **Augmentations:** Horizontal flips, small rotations, scaling  
- **Normalization:** ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`  

### 3. Feature Extraction  
- **Backbone:** `ResNet-50` pretrained on ImageNet  
- **Modification:** Removed final layers → applied **Global Average Pooling (GAP)**  
- **Dimensionality Reduction:** PCA (retain 95% variance or 128 components)  

### 4. Classifier Training  
- **Model:** Support Vector Machine (SVM)  
- **Hyperparameters:**  
  - Kernel = {Linear, RBF}  
  - Regularization `C ∈ {0.01, 0.1, 1, 10, 100}`  
  - GridSearchCV with Stratified 5-fold cross-validation  
- **Optimization:** Multi-class hinge loss with L₂ regularization  

### 5. Evaluation Metrics  
- Overall accuracy  
- Per-class precision, recall, and F₁-score  
- Confusion matrix visualization  
- t-SNE for embedding separability  

---

## 📊 Results

| Metric | Value |
|---------|-------|
| **Overall Test Accuracy** | **58%** |
| **Best Performing Class** | Brutalist |
| **Most Confused Class** | Gothic |

**Observations:**
- The model correctly classified most *Brutalist* and *Colonial* samples.
- *Gothic* images were frequently misclassified—indicating overlapping visual cues or limited samples.
- Confusion analysis suggests that **data imbalance** and **architectural overlap** remain the main challenges.

---

## ⚙️ Implementation Details

- **Frameworks:** PyTorch, scikit-learn, NumPy, OpenCV, Matplotlib  
- **Feature Extractors Tested:** ResNet-50, VGG-16, MobileNetV2  
- **Dimensionality Reduction:** scikit-learn PCA  
- **Classifier:** Linear & RBF SVM (best performer)  
- **Evaluation Tools:** Confusion matrix, t-SNE plot, cross-validation metrics  

---

## 🧩 Key Insights

- **Transfer Learning** offers a strong starting point even with limited domain-specific data.  
- **Feature interpretability** improves when combining deep CNN embeddings with classical classifiers like SVM.  
- **Augmentation** meaningfully boosts generalization when real-world variability (lighting, obstructions, etc.) is limited.  
- **Data imbalance** must be carefully addressed—especially for nuanced categories like Gothic architecture.  

---

## 🚀 Future Improvements

- Fine-tune CNN layers instead of fully freezing weights  
- Expand dataset with more Yale-specific images  
- Apply **data balancing** techniques (e.g., SMOTE or class-weighted loss)  
- Introduce **explainability methods** (Grad-CAM, SHAP) to visualize architectural features influencing predictions  
- Deploy the trained model in a **React-based web demo** or **mobile app** for on-campus use  

---

## 🧭 Setup & Usage

### 1. Clone Repository
```bash
git clone https://github.com/<yourusername>/Architectural-Vision.git
cd Architectural-Vision
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Get the full dataset and preserve folder structure:  
🔗 [Project Data Folder](https://drive.google.com/drive/folders/1ka4ALNQunVQw7EMxK-er9WIi_62UIitJ?usp=sharing)

### 4. Run Pipeline
Execute the notebooks in order:
```
01_build_dataset.ipynb → 02_augment.ipynb → 03_deep_features.ipynb → 04_train_models.ipynb
```

---

## 🧾 References

- Kaggle Architecture Styles Dataset: [wwymak/architecture-dataset](https://www.kaggle.com/datasets/wwymak/architecture-dataset)  
- He, K. et al. *Deep Residual Learning for Image Recognition*, CVPR 2016.  
- Scikit-learn documentation: [https://scikit-learn.org/](https://scikit-learn.org/)  
- PyTorch documentation: [https://pytorch.org/](https://pytorch.org/)

---

## 📸 Sample Results (Optional to Add)

You can add confusion matrices or sample predictions as images, e.g.:

```markdown
![Confusion Matrix](images/confusion_matrix.png)
![t-SNE Visualization](images/tsne_plot.png)
```
