# 🧬 DeepBRCA: Real-Time Machine Learning & Neural Network from Scratch for BRCA1/2 Variant Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Field](https://img.shields.io/badge/Field-Computational_Genomics_%7C_MedAI-brightgreen.svg)]()

An interactive, educational, and benchmarking framework designed to classify functional outcomes of **BRCA1 & BRCA2 missense variants**. This repository bridges molecular oncology and machine learning by demonstrating:
1. **A multi-model clinical benchmarking suite** (Logistic Regression, Random Forest, Multi-Layer Perceptron).
2. **A from-scratch Neural Network (NumPy-only)** with real-time backpropagation visualization and glowing synaptic activation maps.

---

## 📌 Clinical & Biological Rationale

Variants of Uncertain Significance (**VUS**) in *BRCA1* and *BRCA2* leave thousands of oncology patients without actionable preventive or therapeutic strategies (e.g., PARP inhibitors, prophylactic mastectomy). 

This project leverages high-throughput functional data from **Saturation Genome Editing (SGE)** ([Findlay et al., *Nature* 2018](https://doi.org/10.1038/s41586-018-0461-z)) across thousands of assayed SNVs in human breast cells. The dataset is integrated with **25 bioinformatic in-silico scores** (e.g., REVEL, CADD, BayesDel, PolyPhen-2, SIFT) to differentiate **Functionally Damaging** from **Neutral/Benign** variants.

---

## 🚀 Key Modules

### 1. Clinical Evaluation Dashboard (`brca_real_data.py`)
* Benchmarks 3 distinct models on **9,200+ BRCA1 variants**.
* Evaluates using clinical-grade metrics: ROC-AUC, Confusion Matrix, and probabilistic confidence.
* Visualizes live training progression and prediction distributions on held-out test mutations.

### 2. Neural Network from Scratch (`brca_training_visual.py`)
* Implements vectorization, forward pass, multi-class Cross-Entropy Loss, and Backpropagation using **pure NumPy** (no PyTorch/TensorFlow).
* Features a custom dynamic Matplotlib UI displaying:
  * Weight matrix shifts (glow effects indicating excitatory vs. inhibitory weights).
  * Layer-wise activation changes per variant.
  * Real-time loss minimization and accuracy convergence curves.

---

## 📊 Benchmark Results

| Model | Architecture / Hyperparameters | Test Accuracy | ROC-AUC |
| :--- | :--- | :---: | :---: |
| **Random Forest** | 100 Trees, Gini Impurity | **~99.8%** | **~1.000** |
| **Neural Network (MLP)** | Hidden Layers: (64, 32), ReLU, Adam | **~98.1%** | **~0.995** |
| **Logistic Regression** | L2 Regularization, Max Iter: 1000 | **~94.9%** | **~0.985** |

> *Note on Meta-Prediction*: The high discriminatory performance is achieved because features include consensus meta-predictors (such as REVEL and BayesDel) which already integrate structural, conservation, and phylogenetic signals.

---

## 🛠️ Installation & Reproduction

```bash
# 1. Clone repository
git clone https://github.com/<YOUR-USERNAME>/DeepBRCA-Variant-Classification.git
cd DeepBRCA-Variant-Classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Benchmark Dashboard
python src/brca_real_data.py

# 4. Run Interactive Neural Network Animation
python src/brca_training_visual.py
📦 Requirements (requirements.txt)
code
Text
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
🔬 Dataset Acknowledgments
Data curated and processed from the public scientific repository:
Original SGE functional assays: Findlay et al., Nature 2018.
Feature compilation & folds: Steven-N-Hart/BRCA-ML.
⚖️ Medical & Ethical Disclaimer
This repository is developed for academic and research demonstration purposes only. It is not cleared for diagnostic decision-making or primary clinical reporting. Patient interpretation must strictly adhere to certified clinical lab standards (ACMG/AMP guidelines).
