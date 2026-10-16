# 🧬 DeepBRCA: Real-Time Machine Learning & Neural Network from Scratch for BRCA1/2 Variant Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Field](https://img.shields.io/badge/Field-Computational_Genomics_%7C_MedAI-brightgreen.svg)]()
[![Data Source](https://img.shields.io/badge/Data-Findlay%202018%20%7C%20BRCA--ML-orange.svg)](https://github.com/Steven-N-Hart/BRCA-ML)

An interactive, educational, and benchmarking framework designed to classify the functional impact of **BRCA1 & BRCA2 missense variants**. This repository bridges molecular oncology, clinical bioinformatics, and deep learning by providing:
1. **A multi-model clinical benchmarking suite** (Logistic Regression, Random Forest, Multi-Layer Perceptron).
2. **A from-scratch Neural Network (NumPy-only)** with real-time backpropagation visualization and glowing synaptic activation maps.

---

## 📌 Clinical & Biological Rationale

Variants of Uncertain Significance (**VUS**) in *BRCA1* and *BRCA2* represent a significant bottleneck in clinical oncology, leaving patients and oncologists without definitive actionable guidance regarding prophylactic surgery or targeted therapies (e.g., PARP inhibitors).

This project evaluates functional consequences using ground-truth experimental labels derived from **Saturation Genome Editing (SGE)** ([Findlay et al., *Nature* 2018](https://doi.org/10.1038/s41586-018-0461-z)). The dataset is enriched with **25 high-confidence in-silico bioinformatics scores** compiled via **BRCA-ML** ([Hart et al., *Cancer Research* 2019](https://doi.org/10.1158/0008-5472.CAN-18-3821)) to classify variants into **Functionally Damaging** versus **Neutral/Benign**.

---

## 🚀 Key Modules

### 1. Clinical Evaluation Dashboard (`brca_real_data.py`)
* Benchmarks 3 distinct classifiers on **9,200+ BRCA1/2 functional variants**.
* Computes clinical-grade metrics: Confusion Matrix, ROC-AUC curves, and prediction confidence intervals.
* Features a 6-panel real-time Matplotlib dashboard tracking convergence, loss curves, and individual variant predictions.

### 2. Neural Network from Scratch (`brca_training_visual.py`)
* Implements mathematical fundamentals from the ground up: Forward propagation, multi-class Cross-Entropy Loss, and Backpropagation using **pure NumPy**.
* Custom real-time dynamic visualization showing:
  * Excitatory vs. inhibitory synaptic weight adjustments (neon glow effects).
  * Neuron-level activation maps for specific patient mutations.
  * Live accuracy and loss minimization trajectories.

---

## 📊 Benchmark Results

| Model | Architecture / Parameters | Test Accuracy | ROC-AUC |
| :--- | :--- | :---: | :---: |
| **Random Forest** | 100 Estimators, Gini Impurity | **~99.8%** | **~1.000** |
| **Neural Network (MLP)** | Hidden Layers: (64, 32), ReLU, Adam | **~98.1%** | **~0.995** |
| **Logistic Regression** | L2 Regularization, Max Iter: 1000 | **~94.9%** | **~0.985** |

> **Methodological Note on In-Silico Features:** The high discriminatory metrics stem from the integration of consensus meta-predictors (such as REVEL, BayesDel, and MetaLR) which aggregate evolutionary conservation, structural stability, and phylogenetic profiles across the genome.

---

## 📁 Dataset Acquisition

The dataset used in this repository originates from the open-source **BRCA-ML** project:
* **Source Repository:** [Steven-N-Hart/BRCA-ML](https://github.com/Steven-N-Hart/BRCA-ML)
* **Raw CSV Directory:** [BRCA-ML/results](https://github.com/Steven-N-Hart/BRCA-ML/tree/master/results)

To run the project locally, place the 10-fold cross-validation CSV files (`BRCA1_res.fold*.csv` and `BRCA2_res.fold*.csv`) inside a `data/` directory at the project root.

---

## 🛠️ Quickstart

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
```

---

## 📦 Requirements (`requirements.txt`)

```text
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

---

## 📚 Scientific Citations

If you use or build upon this work, please cite the foundational studies:

```bibtex
@article{findlay2018accurate,
  title={Accurate classification of BRCA1 variants with saturation genome editing},
  author={Findlay, Gregory M and Daza, Riza M and Martin, Beth and Zhang, Melissa D and Leith, Alexandra P and Gasperini, Molly and Janizek, Joseph D and Huang, Xiaoyi and Starita, Lea M and Shendure, Jay},
  journal={Nature},
  volume={562},
  number={7726},
  pages={217--222},
  year={2018},
  publisher={Nature Publishing Group},
  doi={10.1038/s41586-018-0461-z}
}

@article{hart2019prediction,
  title={Prediction of the functional impact of missense variants in BRCA1 and BRCA2 with BRCA-ML},
  author={Hart, Steven N and Polley, Eric C and Shimelis, Hermela and Yadav, Siddhartha and Hu, Chunling and Na, Jie and Lee, Kun Y and Couch, Fergus J},
  journal={Cancer Research},
  volume={79},
  number={13},
  pages={3184--3192},
  year={2019},
  publisher={AACR},
  doi={10.1158/0008-5472.CAN-18-3821}
}
```

---

## ⚖️ Medical & Ethical Disclaimer
This repository is strictly for **academic, educational, and research evaluation purposes**. It is not intended or certified for clinical diagnosis, therapeutic decisions, or genetic counseling. Real-world patient variants must be classified under validated clinical laboratory workflows complying with ACMG/AMP clinical genomics guidelines.
