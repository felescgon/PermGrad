# PermGrad: Hybrid Neural Networks with Synthetic Image Representations for Tabular Data

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/felescgon/PermGrad/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Documentation Status](https://readthedocs.org/projects/morph-kgc/badge/?version=latest)](https://tintolib.readthedocs.io/en/latest/)
[![TINTOlib](https://img.shields.io/badge/library-TINTOlib-9cf)](https://github.com/oeg-upm/TINTOlib)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/felescgon/PermGrad)
[![PyPI Downloads](https://static.pepy.tech/badge/tintolib)](https://pepy.tech/projects/tintolib)

This repository contains the implementation, experiments, and evaluation pipeline supporting the scientific article:

**"PermGrad: A Hybrid Neural Network Framework for Global Feature Attribution Using Synthetic Image Representations"**

The project introduces a Hybrid Neural Network (HyNN) that jointly processes:
- **Synthetic images** generated from tabular data using **TINTOlib**, captured by a **CNN branch**.
- **Raw tabular features**, processed by an **MLP branch**.

A unified interpretability mechanism, **PermGrad**, integrates permutation-based importance (MLP), Grad-CAM saliency (CNN), and pruning-based branch contribution weighting to produce *modality-aware*, *global*, and *class-consistent* feature attributions.

---

## 📘 Abstract

Deep learning models struggle to generalise on tabular data due to limited spatial structure and reduced interpretability.  
This work proposes **PermGrad**, a Hybrid Neural Network (HyNN) that unifies symbolic (MLP) and spatial (CNN) reasoning by transforming tabular datasets into **synthetic images** using **TINTO**.  
A three-stage interpretability pipeline—Permutation Importance, Grad-CAM, and pruning-based branch weighting—produces stable, class-consistent global attributions.  
Evaluated across five heterogeneous datasets, the model achieves **competitive predictive performance** while providing **explainable, branch-aware relevance scores** and demonstrating **implicit regularisation** effects in asymmetric settings.

---

## 🖼 Graphical Abstract

The graphical abstract used in the paper is provided under:

```
imgs/graphical_abstract.png
```

<p align="center">
  <img src="imgs/graphical_abstract.png" width="70%">
</p>

---

## 🔎 Explore PermGrad with DeepWiki

PermGrad has a dedicated space on **[DeepWiki](https://deepwiki.com/felescgon/PermGrad)**, where you can explore semantic documentation, relevant links, bibliography, and answers to frequently asked questions about its use and application.

<p align="center">
  <a href="https://deepwiki.com/felescgon/PermGrad" target="_blank">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"/>
  </a>
</p>

---

## 🧩 Synthetic Image Generation Powered by TINTOlib

<p align="center">
  <a href="https://github.com/oeg-upm/TINTOlib" target="_blank">
    <img src="https://raw.githubusercontent.com/oeg-upm/TINTOlib/master/imgs/tintolib_logo.png" 
         alt="TINTOlib Logo" width="240">
  </a>
</p>

This project relies on **TINTOlib**, an open-source Python library for transforming tabular data into  
**synthetic, spatially structured images**.  
TINTOlib provides deterministic, interpretable mappings that enable vision architectures  
(CNNs, ViTs, hybrid models) to operate on non-visual data by leveraging feature–pixel correspondence.

**References & Resources:**

- 📘 Documentation: https://tintolib.readthedocs.io  
- 🐍 PyPI: https://pypi.org/project/tintolib  
- 💻 GitHub: https://github.com/oeg-upm/TINTOlib  

TINTOlib was used in this repository to generate all **synthetic image representations** used for  
training and evaluating the Hybrid Neural Network (HyNN) and PermGrad interpretability framework.

---

## 📂 Repository Structure

```
PermGrad/
│
├── imgs/                   # Graphical abstract and relevance visualisations
├── data/                   # Links or loaders for public datasets
├── models/                 # MLP, CNN, and HyNN implementations (PyTorch)
├── interpretability/       # Permutation, Grad-CAM, and PermGrad integration modules
├── experiments/            # Scripts for benchmarking and evaluation
├── notebooks/              # Jupyter notebooks for analysis and visualisation
├── utils/                  # Auxiliary functions (training loops, metrics...)
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🧪 Datasets

All datasets used in this work are **publicly available**:
- HELOC  
- Dengue  
- Covertype  
- GAS  
- Puma  

Dataset loading utilities are provided in:

```
data/load_datasets.py
```

---

## ⚙️ Training Configuration

### Classical baselines
- Random Forest, LightGBM, XGBoost, Extra Trees, Bagging  
- No heavy hyperparameter tuning (to avoid unfair SOTA-style comparison)

### Deep models
- **Optimiser:** AdamW  
- **Learning rate:** OneCycle schedule  
- **Batch size:** 128  
- **Loss:** Cross-entropy (classification), RMSE (regression)  
- **Training:** 200 epochs  
- **Synthetic images:** 20×20 TINTO maps using TINTOlib defaults

---

## 🔍 Interpretability Pipeline

The PermGrad interpretability stack includes:

1. **Permutation Importance (MLP branch)**  
2. **Grad-CAM saliency (CNN branch)**  
3. **Branch pruning weights**, estimating per-pathway relevance  
4. **PermGrad integration**, yielding global feature relevance

Implementation files:

```
interpretability/permutation.py
interpretability/gradcam.py
interpretability/permgrad.py
```

---

## 📊 Reproducing the Experiments

Run the full workflow:

```
python experiments/run_all.py
```

Or execute components manually:

### Train HyNN
```
python experiments/train_hynn.py --dataset heloc
```

### Compute Permutation Importance
```
python interpretability/permutation.py --dataset heloc
```

### Compute Grad-CAM relevance
```
python interpretability/gradcam.py --dataset heloc
```

### Compute PermGrad integration
```
python interpretability/permgrad.py --dataset heloc
```

---

## 📓 Notebooks

The `notebooks/` directory includes end-to-end examples:

| Notebook | Description |
|---------|-------------|
| `TINTO_visualization.ipynb` | Visualise TINTO-based synthetic images |
| `HyNN_training.ipynb` | Training of MLP, CNN, and HyNN |
| `PermGrad_analysis.ipynb` | Complete interpretability workflow |

---

## 💡 Highlights

- Hybrid architecture combining **symbolic** and **spatial** reasoning for tabular data.
- Synthetic image generation using **TINTOlib**, enabling convolutional processing of non-visual features.
- **PermGrad** unifies permutation and saliency analyses into modality-consistent global attributions.
- Stable, class-invariant relevance across heterogeneous datasets.
- Implicit hybrid regularisation observed even when one branch dominates inference.

---

## 🔧 Installation

```
git clone https://github.com/your-org/PermGrad.git
cd PermGrad
pip install -r requirements.txt
```


## 🛡️ License

TINTOlib is available under the **[Apache License 2.0](https://github.com/felescgon/PermGrad/blob/main/LICENSE)**.

---

## 👥 Authors
- **[Felipe Escalera-González](https://github.com/felescgon)**
- **[Manuel Castillo-Cara](https://github.com/manwestc)**
- **[Mariano Rincón-Zamorano]()**
- **[Luis Orozco-Barbosa]()**

---

## 🏛️ Contributors

<div>
<p align = "center">
<kbd><img src="https://raw.githubusercontent.com/DCY1117/TEMP-Images/refs/heads/main/TINTOlib-images/logo-uned-.jpg" alt="Universidad Nacional de Educación a Distancia" width="231"></kbd> <kbd><img src="https://raw.githubusercontent.com/DCY1117/TEMP-Images/refs/heads/main/TINTOlib-images/logo-uclm.png" alt="Universidad de Castilla-La Mancha" width="115"></kbd> 
</p>
</div>