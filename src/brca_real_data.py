# -*- coding: utf-8 -*-
"""
Live Training and Benchmarking on Real BRCA1/2 Data
Data source: Findlay et al. 2018 (Nature) via Steven-N-Hart/BRCA-ML (Cancer Research 2019)
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import time

np.random.seed(42)

# ============================================================
# 1. LOAD DATASET
# ============================================================
print("=" * 70)
print("  Loading REAL BRCA1/BRCA2 dataset...")
print("=" * 70)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "BRCA*_res.fold*.csv")))

if not csv_files:
    # Try searching in current directory's data folder as fallback
    fallback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    csv_files = sorted(glob.glob(os.path.join(fallback_dir, "BRCA*_res.fold*.csv")))

if not csv_files:
    print("\n⚠️  WARNING: No CSV files found in 'data/' folder!")
    print("   Please download CSVs from Steven-N-Hart/BRCA-ML and place in 'data/'")
    print("   Falling back to simulated data for pipeline demonstration...\n")
    from sklearn.datasets import make_classification
    X_sim, y_sim = make_classification(n_samples=5000, n_features=25,
                                        n_informative=15, random_state=42)
    df = pd.DataFrame(X_sim, columns=[f'feat_{i}' for i in range(25)])
    df['predict'] = np.where(y_sim == 1, 'Damaging', 'Neutral')
    data_source = "SIMULATED (Demo Mode)"
else:
    print(f"  Found {len(csv_files)} cross-validation CSV files.")
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    data_source = "REAL (Findlay et al. 2018 / Hart et al. 2019)"

print(f"  Data source: {data_source}")
print(f"  Total variants: {len(df):,}")
print(f"\n  Class distribution:")
print(df['predict'].value_counts())

# ============================================================
# 2. FEATURE EXTRACTION & PREPROCESSING (NO DATA LEAKAGE)
# ============================================================
FEATURE_COLS = [
    'RevelScore', 'CaddRaw', 'BayesDel', 'MetalrScore', 'MetasvmScore',
    'DannScore', 'MCapScore', 'ProveanScore', 'SiftScore',
    'Polyphen2HdivScore', 'Polyphen2HvarScore', 'MutationtasterScore',
    'MutpredScore', 'LrtScore', 'FathmmScore', 'FathmmMklCodingScore',
    'Vest3Score', 'GenocanyonScore', 'EigenRaw', 'EigenPcRaw',
    'AlignGVGDPrior', 'H1HescFitconsScore', 'Gm12878FitconsScore',
    'HuvecFitconsScore', 'IntegratedFitconsScore',
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
if not FEATURE_COLS:
    FEATURE_COLS = [c for c in df.columns if c.startswith('feat_')]

print(f"\n  ✅ Features detected: {len(FEATURE_COLS)}")

# Convert targets to binary labels (1: Damaging, 0: Neutral)
df['label'] = (df['predict'] == 'Damaging').astype(int)

# Convert features to numeric
for col in FEATURE_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_clean = df.dropna(subset=['label']).copy()
print(f"  Valid samples after target filtering: {len(df_clean):,}")

X = df_clean[FEATURE_COLS].values
y = df_clean['label'].values

# Split data first to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Impute missing values based strictly on training distributions
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# Standardize features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_imp)
X_test_s = scaler.transform(X_test_imp)

print(f"  Training set: {len(X_train):,}  |  Testing set: {len(X_test):,}")

# ============================================================
# 3. BENCHMARK MODELS
# ============================================================
print("\n  Training baseline classifiers...")

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_s, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_s))

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test_s))

nn = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                    solver='adam', max_iter=1, warm_start=True,
                    random_state=42)
train_loss = []
epochs = 40

# ============================================================
# 4. DASHBOARD INITIALIZATION
# ============================================================
plt.ion()
fig = plt.figure(figsize=(20, 12))
fig.suptitle(f'Live AI Clinical Benchmark on Real BRCA Data  —  Source: {data_source}',
             fontsize=17, fontweight='bold', y=0.98)

gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                       left=0.06, right=0.97, top=0.90, bottom=0.07)

# Panel 1: Loss
ax1 = fig.add_subplot(gs[0, 0])
line_loss, = ax1.plot([], [], color='#e74c3c', linewidth=2.5)
fill_loss = None
ax1.set_xlim(0, epochs); ax1.set_ylim(0, 1)
ax1.set_xlabel('Epoch', fontsize=11); ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('1) Convergence Curve (MLP)', fontsize=13, fontweight='bold',
              color='#c0392b', pad=10)
ax1.grid(alpha=0.3); ax1.set_axisbelow(True); ax1.tick_params(labelsize=10)

# Panel 2: Accuracy
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(-0.5, 2.5); ax2.set_ylim(50, 100)
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(['Logistic Reg', 'Random Forest', 'Neural Net'], fontsize=10)
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.set_title('2) Model Benchmark Comparison', fontsize=13, fontweight='bold',
              color='#c0392b', pad=10)
ax2.grid(axis='y', alpha=0.3); ax2.set_axisbelow(True); ax2.tick_params(labelsize=10)
bar_lr = ax2.bar(0, lr_acc*100, width=0.6, color='#3498db', edgecolor='black', alpha=0.9)
bar_rf = ax2.bar(1, rf_acc*100, width=0.6, color='#2ecc71', edgecolor='black', alpha=0.9)
bar_nn = ax2.bar(2, 0, width=0.6, color='#e74c3c', edgecolor='black', alpha=0.9)
text_lr = ax2.text(0, lr_acc*100 + 1, f'{lr_acc*100:.1f}%', ha='center',
                    fontsize=11, fontweight='bold')
text_rf = ax2.text(1, rf_acc*100 + 1, f'{rf_acc*100:.1f}%', ha='center',
                    fontsize=11, fontweight='bold')
text_nn = ax2.text(2, 52, '', ha='center', fontsize=11, fontweight='bold')
ax2.axhline(y=95, color='gray', linestyle='--', alpha=0.6)

# Panel 3: Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_title('3) Confusion Matrix (Test Set)', fontsize=13, fontweight='bold', pad=10)
ax3.text(0.5, 0.5, 'Training in progress...', ha='center', va='center',
         fontsize=11, color='gray', transform=ax3.transAxes)
ax3.set_xticks([]); ax3.set_yticks([])

# Panel 4: ROC Curves
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_title('4) ROC Curves & Discrimination', fontsize=13, fontweight='bold', pad=10)
ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
ax4.text(0.5, 0.5, 'Training in progress...', ha='center', va='center',
         fontsize=11, color='gray', transform=ax4.transAxes)
ax4.set_xticks([]); ax4.set_yticks([])

# Panel 5: Test Case Predictions
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_title('5) Held-out Variant Predictions',
              fontsize=13, fontweight='bold', pad=10)
ax5.text(0.5, 0.5, 'Waiting for model...', ha='center', va='center',
         fontsize=11, color='gray', transform=ax5.transAxes)
ax5.set_xticks([]); ax5.set_yticks([])

# Panel 6: Live Status
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off'); ax6.set_xlim(0, 1); ax6.set_ylim(0, 1)
ax6.set_title('6) Live Telemetry', fontsize=13, fontweight='bold', pad=10)

card_epoch = FancyBboxPatch((0.03, 0.72), 0.44, 0.20,
                             boxstyle="round,pad=0.02", facecolor='#3498db',
                             edgecolor='black', alpha=0.9)
ax6.add_patch(card_epoch)
ax6.text(0.25, 0.87, 'EPOCH', ha='center', fontsize=10, color='white', fontweight='bold')
text_epoch_val = ax6.text(0.25, 0.78, f'0 / {epochs}', ha='center',
                           fontsize=14, color='white', fontweight='bold')

card_loss = FancyBboxPatch((0.53, 0.72), 0.44, 0.20,
                            boxstyle="round,pad=0.02", facecolor='#e74c3c',
                            edgecolor='black', alpha=0.9)
ax6.add_patch(card_loss)
ax6.text(0.75, 0.87, 'LOSS', ha='center', fontsize=10, color='white', fontweight='bold')
text_loss_val = ax6.text(0.75, 0.78, '---', ha='center', fontsize=14,
                          color='white', fontweight='bold')

card_acc = FancyBboxPatch((0.03, 0.43), 0.44, 0.20,
                           boxstyle="round,pad=0.02", facecolor='#27ae60',
                           edgecolor='black', alpha=0.9)
ax6.add_patch(card_acc)
ax6.text(0.25, 0.58, 'ACCURACY', ha='center', fontsize=10, color='white', fontweight='bold')
text_acc_val = ax6.text(0.25, 0.49, '---', ha='center', fontsize=14,
                         color='white', fontweight='bold')

card_time = FancyBboxPatch((0.53, 0.43), 0.44, 0.20,
                            boxstyle="round,pad=0.02", facecolor='#9b59b6',
                            edgecolor='black', alpha=0.9)
ax6.add_patch(card_time)
ax6.text(0.75, 0.58, 'TIME', ha='center', fontsize=10, color='white', fontweight='bold')
text_time_val = ax6.text(0.75, 0.49, '0.0s', ha='center', fontsize=14,
                          color='white', fontweight='bold')

text_status = ax6.text(0.5, 0.26, 'Optimizing Neural Network...',
                       ha='center', fontsize=11, fontweight='bold', color='#c0392b')
ax6.text(0.5, 0.08, f'Dataset: {len(df_clean):,} annotated variants',
         ha='center', fontsize=10, color='gray', style='italic')

fig.canvas.draw(); fig.canvas.flush_events()

# ============================================================
# 5. LIVE NEURAL NET TRAINING
# ============================================================
start_time = time.time()

for epoch in range(epochs):
    nn.fit(X_train_s, y_train)
    train_loss.append(nn.loss_)
    current_acc = accuracy_score(y_test, nn.predict(X_test_s)) * 100

    line_loss.set_data(range(1, len(train_loss) + 1), train_loss)
    if fill_loss is not None:
        fill_loss.remove()
    fill_loss = ax1.fill_between(range(1, len(train_loss) + 1),
                                  train_loss, alpha=0.2, color='#e74c3c')
    ax1.set_ylim(0, max(train_loss) * 1.1)

    bar_nn[0].set_height(current_acc)
    text_nn.set_text(f'{current_acc:.1f}%'); text_nn.set_y(current_acc + 1)

    text_epoch_val.set_text(f'{epoch+1} / {epochs}')
    text_loss_val.set_text(f'{nn.loss_:.3f}')
    text_acc_val.set_text(f'{current_acc:.1f}%')
    text_time_val.set_text(f'{time.time() - start_time:.1f}s')

    if nn.loss_ < 0.2:
        card_loss.set_facecolor('#27ae60')
    elif nn.loss_ < 0.4:
        card_loss.set_facecolor('#f39c12')

    fig.canvas.draw(); fig.canvas.flush_events()
    plt.pause(0.08)

nn_acc = accuracy_score(y_test, nn.predict(X_test_s)) * 100
text_status.set_text('Training Complete!')
text_status.set_color('#27ae60')
fig.canvas.draw(); fig.canvas.flush_events()
plt.pause(0.6)

# ============================================================
# 6. POPULATE METRICS PANELS
# ============================================================

# Confusion Matrix
ax3.clear()
cm = confusion_matrix(y_test, nn.predict(X_test_s))
ax3.imshow(cm, cmap='Blues', aspect='auto')
ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Neutral', 'Damaging'], fontsize=10)
ax3.set_yticklabels(['Neutral', 'Damaging'], fontsize=10)
ax3.set_xlabel('Predicted Label', fontsize=10)
ax3.set_ylabel('Experimental Label (SGE)', fontsize=10)
ax3.set_title('3) Confusion Matrix (MLP)', fontsize=13, fontweight='bold', pad=10)
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(cm[i, j]), ha='center', va='center',
                 fontsize=18, fontweight='bold',
                 color='white' if cm[i, j] > cm.max()/2 else 'black')

# ROC Curves
ax4.clear()
for model, name, color in [(lr, 'Logistic Regression', '#3498db'),
                            (rf, 'Random Forest', '#2ecc71'),
                            (nn, 'Neural Network', '#e74c3c')]:
    prob = model.predict_proba(X_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    ax4.plot(fpr, tpr, color=color, linewidth=2.5,
             label=f'{name} (AUC={roc_auc:.3f})')
ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')
ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
ax4.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
ax4.set_ylabel('True Positive Rate (Sensitivity)', fontsize=10)
ax4.set_title('4) ROC Discrimination Curves', fontsize=13, fontweight='bold', pad=10)
ax4.legend(loc='lower right', fontsize=9)
ax4.grid(alpha=0.3); ax4.tick_params(labelsize=10)

# Sample Predictions
ax5.clear()
sample_idx = np.random.choice(len(X_test), 10, replace=False)
sample_X = X_test_s[sample_idx]
sample_y = y_test[sample_idx]
sample_prob = nn.predict_proba(sample_X)[:, 1]

if 'Variant' in df_clean.columns:
    sample_names = [f"V{i+1}: {df_clean.iloc[idx]['Variant'][:18]}" 
                    for i, idx in enumerate(sample_idx)]
else:
    sample_names = [f"Variant #{idx}" for idx in sample_idx]

color_map_real = {0: "#27ae60", 1: "#e74c3c"}
bar_colors = [color_map_real[int(lbl)] for lbl in sample_y]

y_pos = np.arange(len(sample_names))
bars = ax5.barh(y_pos, sample_prob, color=bar_colors,
                edgecolor='black', alpha=0.9, height=0.7)
ax5.set_yticks(y_pos)
ax5.set_yticklabels(sample_names, fontsize=8)
ax5.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
for bar, prob in zip(bars, sample_prob):
    ax5.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
             f'{prob*100:.0f}%', va='center', fontsize=9, fontweight='bold')
ax5.set_xlim(0, 1.15)
ax5.set_xlabel('Predicted Probability of Pathogenicity', fontsize=10)
ax5.set_title('5) Sample Predictions on Held-Out Variants\n(Red=True Damaging, Green=True Neutral)',
              fontsize=11, fontweight='bold', pad=10)
ax5.grid(axis='x', alpha=0.3); ax5.set_axisbelow(True)
ax5.tick_params(axis='x', labelsize=10)

fig.canvas.draw(); fig.canvas.flush_events()

# ============================================================
# 7. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("  FINAL PERFORMANCE SUMMARY")
print("=" * 70)
print(f"  Data source:          {data_source}")
print(f"  Total analyzed:       {len(df_clean):,}")
print(f"  Features used:        {len(FEATURE_COLS)}")
print(f"  Train / Test size:    {len(X_train):,} / {len(X_test):,}")
print("-" * 70)
print(f"  Logistic Regression:  {lr_acc*100:.2f}%")
print(f"  Random Forest:        {rf_acc*100:.2f}%")
print(f"  Neural Network (MLP): {nn_acc:.2f}%")
print(f"  Elapsed Time:         {time.time() - start_time:.1f}s")
print("=" * 70)

plt.ioff()
plt.show()
