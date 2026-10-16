# -*- coding: utf-8 -*-
"""
Visual Neural Network Training from Scratch for BRCA1/BRCA2
Implementation of Forward Propagation & Backpropagation in Pure NumPy
Interactive Dark-Themed Live Synaptic Activation & Weight Glow Dashboard
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ============================================================
# 1. LOAD REAL BRCA DATASET
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Search for CSV files in project directories
files = sorted(glob.glob(os.path.join(BASE_DIR, "..", "data", "BRCA*_res.fold*.csv")))
if not files:
    files = sorted(glob.glob(os.path.join(BASE_DIR, "**", "BRCA*_res.fold*.csv"), recursive=True))

FEATURES = ['RevelScore', 'CaddRaw', 'BayesDel', 'MetalrScore',
            'Polyphen2HdivScore', 'ProveanScore', 'SiftScore', 'Vest3Score']
FEATURE_SHORT = ['REVEL', 'CADD', 'BayesDel', 'MetaLR',
                 'PolyPhen2', 'PROVEAN', 'SIFT', 'VEST3']

if files:
    dfs = [pd.read_csv(f) for f in files[:5]]
    df = pd.concat(dfs, ignore_index=True)
    df['label'] = (df['predict'] == 'Damaging').astype(int)
    
    # Process selected features
    available_features = [c for c in FEATURES if c in df.columns]
    for c in available_features:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median())
        
    df = df.dropna(subset=['label']).sample(n=min(3000, len(df)), random_state=42)
    X = df[available_features].values
    y = df['label'].values
    print(f"Loaded {len(df):,} real BRCA1 variant samples.")
else:
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=3000, n_features=8, n_informative=6,
                                n_classes=2, random_state=42)
    print("⚠️ WARNING: Using simulated data (no CSV files found in 'data/' folder)")

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 2. NEURAL NETWORK IMPLEMENTED FROM SCRATCH (NUMPY ONLY)
# ============================================================
class MLP:
    def __init__(self, sizes):
        """
        Custom Multi-Layer Perceptron using He Initialization.
        """
        self.sizes = sizes
        self.W = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0/sizes[i])
                  for i in range(len(sizes)-1)]
        self.b = [np.zeros((1, sizes[i+1])) for i in range(len(sizes)-1)]

    def forward(self, X):
        """Forward pass with ReLU for hidden layers and Softmax for output."""
        self.a = [X]
        self.z = []
        cur = X
        for i in range(len(self.W)):
            z = cur @ self.W[i] + self.b[i]
            self.z.append(z)
            if i < len(self.W) - 1:
                cur = np.maximum(0, z)  # ReLU Activation
            else:
                e = np.exp(z - z.max(axis=1, keepdims=True))
                cur = e / e.sum(axis=1, keepdims=True)  # Stable Softmax
            self.a.append(cur)
        return self.a[-1]

    def loss(self, probs, y):
        """Categorical Cross-Entropy Loss."""
        m = len(y)
        return -np.log(probs[np.arange(m), y] + 1e-10).mean()

    def backward(self, X, y, lr=0.5):
        """Backpropagation with Stochastic Gradient Descent (SGD)."""
        m = X.shape[0]
        y_oh = np.zeros((m, self.sizes[-1]))
        y_oh[np.arange(m), y] = 1
        dz = self.a[-1] - y_oh
        for i in range(len(self.W)-1, -1, -1):
            dW = self.a[i].T @ dz / m
            db = dz.sum(axis=0, keepdims=True) / m
            if i > 0:
                da = dz @ self.W[i].T
                dz = da * (self.z[i-1] > 0)  # Derivative of ReLU
            self.W[i] -= lr * dW
            self.b[i] -= lr * db

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


layer_sizes = [8, 8, 6, 2]
net = MLP(layer_sizes)

# ============================================================
# 3. VISUALIZATION CANVAS (NEON GLOW + TELEMETRY)
# ============================================================
plt.style.use('dark_background')
plt.ion()
fig = plt.figure(figsize=(18, 10), facecolor='#0d1117')

fig.suptitle('Neural Network Mathematical Learning on Real BRCA1 Data',
             fontsize=16, fontweight='bold', y=1.00, color='#00e5ff')
fig.text(0.5, 0.965,
         'Real-Time Synaptic Adjustments, Forward Pass & Backpropagation Convergence',
         ha='center', fontsize=11, style='italic', color='#8b949e')

# --- Progress Bar ---
ax_prog = fig.add_axes([0.68, 0.92, 0.29, 0.035])
ax_prog.set_xlim(0, 1); ax_prog.set_ylim(0, 1); ax_prog.axis('off')
prog_bg = Rectangle((0, 0), 1, 1, facecolor='#21262d',
                     edgecolor='#30363d', linewidth=1)
ax_prog.add_patch(prog_bg)
prog_bar = Rectangle((0, 0), 0, 1, facecolor='#00e5ff', edgecolor='none')
ax_prog.add_patch(prog_bar)
prog_text = ax_prog.text(0.5, 0.5, '0%', ha='center', va='center',
                          fontsize=10, fontweight='bold', color='#ffffff')

# --- Network Topology (Left) ---
ax_net = fig.add_axes([0.03, 0.10, 0.58, 0.78], facecolor='#0d1117')
ax_net.set_xlim(-1.2, len(layer_sizes) - 0.1)
ax_net.set_ylim(-max(layer_sizes)/2 - 2.5, max(layer_sizes)/2 + 0.8)
ax_net.axis('off')

# --- Loss Convergence (Top-Right) ---
ax_loss = fig.add_axes([0.68, 0.55, 0.29, 0.33], facecolor='#161b22')
ax_loss.set_title('Training Cross-Entropy Loss', fontsize=12, fontweight='bold',
                  color='#ff7b72', pad=10)
ax_loss.set_xlabel('Epoch', fontsize=10, color='#8b949e')
ax_loss.set_ylabel('Loss', fontsize=10, color='#8b949e')
ax_loss.grid(alpha=0.2, color='#30363d')
ax_loss.set_axisbelow(True)
ax_loss.tick_params(colors='#8b949e', labelsize=9)

# --- Generalization Accuracy (Bottom-Right) ---
ax_acc = fig.add_axes([0.68, 0.10, 0.29, 0.33], facecolor='#161b22')
ax_acc.set_title('Held-Out Test Accuracy', fontsize=12, fontweight='bold',
                 color='#7ee787', pad=10)
ax_acc.set_xlabel('Epoch', fontsize=10, color='#8b949e')
ax_acc.set_ylabel('Accuracy (%)', fontsize=10, color='#8b949e')
ax_acc.grid(alpha=0.2, color='#30363d')
ax_acc.set_axisbelow(True)
ax_acc.set_ylim(50, 100)
ax_acc.tick_params(colors='#8b949e', labelsize=9)

# Node Coordinate Map
node_pos = []
for l, size in enumerate(layer_sizes):
    ys = np.arange(size) - (size - 1) / 2
    node_pos.append([(l, y) for y in ys])

layer_labels = ['Input Layer\n(8 Features)', 'Hidden Layer 1\n(8 Neurons)',
                'Hidden Layer 2\n(6 Neurons)', 'Output Layer\n(2 Classes)']

loss_history, acc_history = [], []

# Glow effect lines
for lw, al in [(8, 0.10), (5, 0.25), (2.5, 1.0)]:
    ax_loss.plot([], [], color='#ff7b72', linewidth=lw, alpha=al)
line_loss, = ax_loss.plot([], [], color='#ff7b72', linewidth=2.5)

for lw, al in [(8, 0.10), (5, 0.25), (2.5, 1.0)]:
    ax_acc.plot([], [], color='#7ee787', linewidth=lw, alpha=al)
line_acc, = ax_acc.plot([], [], color='#7ee787', linewidth=2.5)


# ============================================================
# 4. DYNAMIC PLOTTING UTILITY
# ============================================================
def draw_network(activations=None, prediction=None, true_label=None,
                 epoch=0, current_loss=0, current_acc=0):
    ax_net.clear()
    ax_net.set_xlim(-1.2, len(layer_sizes) - 0.1)
    ax_net.set_ylim(-max(layer_sizes)/2 - 2.5, max(layer_sizes)/2 + 0.8)
    ax_net.axis('off')
    ax_net.set_title(f'Forward Propagation State — Epoch {epoch} / 40',
                     fontsize=13, fontweight='bold', pad=14, color='#00e5ff')

    # Layer Headers
    for l, label in enumerate(layer_labels):
        ax_net.text(l, -max(layer_sizes)/2 - 0.9, label,
                    ha='center', va='top', fontsize=10,
                    fontweight='bold', color='#c9d1d9')

    # Biological In-Silico Feature Labels
    if activations is not None:
        for i, name in enumerate(FEATURE_SHORT):
            x, y = node_pos[0][i]
            ax_net.text(x - 0.45, y, name, ha='right', va='center',
                        fontsize=9, color='#8b949e', fontweight='bold')

    # Output Labels
    n_layers = len(layer_sizes)
    ax_net.text(n_layers - 1 + 0.55, -0.3, 'Neutral (0)',
                ha='left', va='center', fontsize=11,
                color='#7ee787', fontweight='bold')
    ax_net.text(n_layers - 1 + 0.55, 0.3, 'Damaging (1)',
                ha='left', va='center', fontsize=11,
                color='#ff7b72', fontweight='bold')

    # Synaptic Weights with Glow Mapping
    for l in range(len(layer_sizes) - 1):
        W = net.W[l]
        w_max = max(abs(W).max(), 0.1)
        for i, (x1, y1) in enumerate(node_pos[l]):
            for j, (x2, y2) in enumerate(node_pos[l+1]):
                w = W[i, j]
                color = '#58a6ff' if w > 0 else '#ff7b72'
                base_alpha = min(abs(w) / w_max, 0.5)
                # Neon glow layers
                for lw, al in [(4, base_alpha*0.15), (2, base_alpha*0.35)]:
                    ax_net.plot([x1, x2], [y1, y2], color=color,
                                alpha=al, linewidth=lw, zorder=1)
                # Primary connection
                ax_net.plot([x1, x2], [y1, y2], color=color,
                            alpha=base_alpha,
                            linewidth=0.5 + base_alpha*2, zorder=2)

    # Neurons / Activations
    for l, positions in enumerate(node_pos):
        for i, (x, y) in enumerate(positions):
            if activations is not None and l < len(activations):
                val = activations[l][0, i]
                if l == 0:
                    intensity = min(abs(val) / 3.0, 1.0)
                    color = plt.cm.Blues(0.3 + intensity * 0.7)
                else:
                    intensity = min(abs(val), 1.0)
                    color = (plt.cm.Reds(0.15 + intensity * 0.85)
                             if val > 0 else '#21262d')
            else:
                color = '#21262d'

            # Halo
            ax_net.add_patch(Circle((x, y), 0.32, facecolor=color,
                                     edgecolor='none', alpha=0.15, zorder=1))
            # Neuron Body
            ax_net.add_patch(Circle((x, y), 0.28, facecolor=color,
                                     edgecolor='#30363d', linewidth=1.5, zorder=2))

            if activations is not None and l > 0 and l < len(activations):
                val = activations[l][0, i]
                ax_net.text(x, y, f'{val:.2f}', ha='center', va='center',
                            fontsize=7.5, fontweight='bold', zorder=3,
                            color='#ffffff' if val > 0.5 else '#c9d1d9')

    # Telemetry Panel
    if prediction is not None:
        pred_text = 'Damaging' if prediction == 1 else 'Neutral'
        pred_color = '#ff7b72' if prediction == 1 else '#7ee787'
        true_text = 'Damaging' if true_label == 1 else 'Neutral'
        correct = (prediction == true_label)
        status_text = 'CORRECT' if correct else 'INCORRECT'
        status_color = '#7ee787' if correct else '#ff7b72'

        box = FancyBboxPatch((-1.1, -max(layer_sizes)/2 - 2.3), 5.4, 0.7,
                              boxstyle="round,pad=0.05",
                              facecolor='#161b22', edgecolor='#30363d',
                              linewidth=1.5, zorder=0)
        ax_net.add_patch(box)

        ax_net.text(-1.0, -max(layer_sizes)/2 - 1.95, 'Inference:',
                    ha='left', va='center', fontsize=10,
                    color='#c9d1d9', fontweight='bold')
        ax_net.text(-0.35, -max(layer_sizes)/2 - 1.95, pred_text,
                    ha='left', va='center', fontsize=11,
                    color=pred_color, fontweight='bold')
        ax_net.text(0.55, -max(layer_sizes)/2 - 1.95, '| Ground Truth:',
                    ha='left', va='center', fontsize=10,
                    color='#c9d1d9', fontweight='bold')
        ax_net.text(1.45, -max(layer_sizes)/2 - 1.95, true_text,
                    ha='left', va='center', fontsize=11,
                    color='#c9d1d9', fontweight='bold')
        ax_net.text(2.35, -max(layer_sizes)/2 - 1.95, f'[{status_text}]',
                    ha='left', va='center', fontsize=11,
                    color=status_color, fontweight='bold')

    # Legend
    ax_net.plot([], [], color='#58a6ff', linewidth=2.5, label='Positive Weight (+)')
    ax_net.plot([], [], color='#ff7b72', linewidth=2.5, label='Negative Weight (-)')
    ax_net.legend(loc='upper right', fontsize=9, framealpha=0.3,
                  facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='#c9d1d9')


# Initial frame
draw_network()
fig.canvas.draw()
fig.canvas.flush_events()
plt.pause(0.5)

# ============================================================
# 5. TRAINING LOOP
# ============================================================
epochs = 40
batch_size = 32
n_batches = len(X_train) // batch_size

for epoch in range(epochs):
    # Mini-batch shuffling
    idx = np.random.permutation(len(X_train))
    X_shuf, y_shuf = X_train[idx], y_train[idx]

    epoch_loss = 0
    for b in range(n_batches):
        Xb = X_shuf[b*batch_size:(b+1)*batch_size]
        yb = y_shuf[b*batch_size:(b+1)*batch_size]
        probs = net.forward(Xb)
        epoch_loss += net.loss(probs, yb)
        net.backward(Xb, yb, lr=0.5)
    epoch_loss /= n_batches
    loss_history.append(epoch_loss)

    # Validation evaluation
    preds = net.predict(X_test)
    acc = (preds == y_test).mean() * 100
    acc_history.append(acc)

    # Dynamic single-mutation visualization
    si = np.random.randint(0, len(X_test))
    sample = X_test[si:si+1]
    probs_sample = net.forward(sample)
    pred = probs_sample.argmax(axis=1)[0]

    # Render frame
    draw_network(activations=net.a, prediction=pred,
                 true_label=y_test[si], epoch=epoch+1,
                 current_loss=epoch_loss, current_acc=acc)

    # Update metric tracking
    line_loss.set_data(range(1, len(loss_history)+1), loss_history)
    ax_loss.relim()
    ax_loss.autoscale_view()
    line_acc.set_data(range(1, len(acc_history)+1), acc_history)
    ax_acc.relim()
    ax_acc.autoscale_view()

    # Update progress bar
    progress = (epoch + 1) / epochs
    prog_bar.set_bounds(0, 0, progress, 1)
    prog_text.set_text(f'{int(progress * 100)}%')

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.12)

# ============================================================
# 6. SUMMARY
# ============================================================
final_preds = net.predict(X_test)
final_acc = (final_preds == y_test).mean() * 100
print("\n" + "=" * 60)
print("  TRAINING CONVERGENCE SUMMARY")
print("=" * 60)
print(f"  Final Held-Out Accuracy: {final_acc:.2f}%")
print(f"  Final Cross-Entropy Loss: {loss_history[-1]:.4f}")
print(f"  Total Epochs Completed:   {epochs}")
print("=" * 60)

plt.ioff()
plt.show()
