# 📂 Dataset & Folders Guide

To run the pipeline on the full experimental dataset (Findlay et al., 2018 / Hart et al., 2019):

### 1. Download the Data
Download the cross-validation CSV files directly from the official **BRCA-ML** repository:
* 🔗 [BRCA-ML Results Directory](https://github.com/Steven-N-Hart/BRCA-ML/tree/master/results)
* 📦 [Direct ZIP Download](https://github.com/Steven-N-Hart/BRCA-ML/archive/refs/heads/master.zip)

### 2. File Structure
Extract and place the 10 cross-validation fold files inside this directory (`data/`):
```text
data/
├── BRCA1_res.fold1.csv
├── BRCA1_res.fold2.csv
├── ...
├── BRCA1_res.fold10.csv
├── BRCA2_res.fold1.csv
└── ...
```

> **Automated Fallback:** If these CSV files are not detected when running the scripts, the system will automatically initialize in **Simulated Demo Mode** using high-dimensional synthetic distributions to ensure reproducible pipeline execution.
