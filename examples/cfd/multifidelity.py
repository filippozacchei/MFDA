"""
Train Multi-Fidelity Neural Networks for BFS Dataset
====================================================

Trains two MultiFidelityNN models:

1. LF → HF      : maps low-fidelity outlet profiles + parameters to high-fidelity
2. LF + MF → HF : maps low- and mid-fidelity outlet profiles + parameters to high-fidelity

Each sample corresponds to an outlet velocity profile (100 points).
Two CMAME-ready figures are generated:
  • RMS scatter (reconstruction accuracy)
  • Representative profile reconstruction
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Add model path
# ----------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/forward_models"))
from multi_fidelity_nn import MultiFidelityNN

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
DATA_PATH = Path("data_bfs_multifidelity/dataset.csv")
MODEL_DIR = Path("models/bfs_multifidelity")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
n_points = 100

# ----------------------------------------------------
# Load and preprocess dataset
# ----------------------------------------------------
df = pd.read_csv(DATA_PATH)
fmap = {"LF": 0, "MF": 1, "HF": 2}
df["fid"] = df["fidelity"].map(fmap)

def to_profiles(subdf):
    grouped = subdf.groupby(["H_in", "U_in"])
    X, Y = [], []
    for (H, U), g in grouped:
        g = g.sort_values("y")
        if len(g) == n_points:
            X.append([H, U])
            Y.append(g["u"].values)
    return np.array(X), np.array(Y)

X_LF, Y_LF = to_profiles(df[df.fid == 0])
X_MF, Y_MF = to_profiles(df[df.fid == 1])
X_HF, Y_HF = to_profiles(df[df.fid == 2])

# ----------------------------------------------------
# Align by nearest parameters (robust)
# ----------------------------------------------------
def align_nearest(X_ref, Y_ref, X_tar, Y_tar, tol=1e-5):
    X_ref, Y_ref = np.array(X_ref), np.array(Y_ref)
    X_tar, Y_tar = np.array(X_tar), np.array(Y_tar)
    Y_tar_aligned = []
    for x in X_ref:
        d = np.linalg.norm(X_tar - x, axis=1)
        j = np.argmin(d)
        if d[j] < tol:
            Y_tar_aligned.append(Y_tar[j])
        else:
            Y_tar_aligned.append(np.full_like(Y_ref[0], np.nan))
    return X_ref, Y_ref, np.array(Y_tar_aligned)

X_HF, Y_HF, Y_LF_al = align_nearest(X_HF, Y_HF, X_LF, Y_LF)
_, _, Y_MF_al = align_nearest(X_HF, Y_HF, X_MF, Y_MF)

mask = ~np.isnan(Y_LF_al).any(axis=1) & ~np.isnan(Y_MF_al).any(axis=1)
X_HF, Y_HF, Y_LF_al, Y_MF_al = X_HF[mask], Y_HF[mask], Y_LF_al[mask], Y_MF_al[mask]
logging.info(f"Aligned {len(X_HF)} consistent samples across fidelities.")

# ----------------------------------------------------
# Split into training and test sets
# ----------------------------------------------------
X_train, X_test, Y_train_LF, Y_test_LF, Y_train_MF, Y_test_MF, Y_train_HF, Y_test_HF = \
    train_test_split(X_HF, Y_LF_al, Y_MF_al, Y_HF, test_size=0.1, random_state=42)

# ----------------------------------------------------
# Neural network configurations
# ----------------------------------------------------
# ----------------------------------------------------
# Neural network configurations
# ----------------------------------------------------
layers_config_1 = {
    "input_layers": {
        "param": [
            {"units": 32, "activation": "gelu", "rate": 0.0},
            {"units": 32, "activation": "gelu", "rate": 0.0},
            {"units": 32, "activation": "gelu", "rate": 0.0},
            {"units": 32, "activation": "gelu", "rate": 0.0},
            {"units": 32, "activation": "linear", "rate": 0.0},
        ],
        "coarse_solution1": [
            {"units": 32, "activation": "linear", "rate": 0.0},
        ],
    },
    "output_layers": [
        {"units": 64, "activation": "gelu", "rate": 0.0},
        {"units": n_points, "activation": "linear", "rate": 0.0},
    ],
}

layers_config_2 = {
    "input_layers": {
        "param": [
            {"units": 32, "activation": "gelu", "rate": 0.0},
            {"units": 32, "activation": "gelu", "rate": 0.0},
            {"units": 32, "activation": "gelu", "rate": 0.0},
            {"units": 32, "activation": "gelu", "rate": 0.0},
            {"units": 32, "activation": "linear", "rate": 0.0},
        ],
        "coarse_solution1": [
            {"units": 32, "activation": "linear", "rate": 0.0},
        ],
        "coarse_solution2": [
            {"units": 32, "activation": "linear", "rate": 0.0},
        ],
    },
    "output_layers": [
        {"units": 64, "activation": "gelu", "rate": 0.0},
        {"units": n_points, "activation": "linear", "rate": 0.0},
    ],
}

train_config = {
    "epochs": 500,
    "batch_size": 8,
    "n_splits": 1,
    "optimizer": "adam",
    "scheduler_coeff": 0.99,
    "scheduler_mode": "linear",
    "step": 10,
    "model_save_path": str(MODEL_DIR),
}

# ----------------------------------------------------
# Train LF → HF
# ----------------------------------------------------
logging.info("Training model: LF → HF")

model_LF_HF = MultiFidelityNN(
    input_shapes=[(2,), (n_points,)],
    coeff=1e-8,
    layers_config=layers_config_1,
    train_config=train_config,
    output_units=n_points,
    output_activation="linear",
    merge_mode="concat",
)

model_LF_HF.train(
    X_train_fidelities=[X_train, Y_train_LF],
    y_train=Y_train_HF,
    X_test_fidelities=[X_test, Y_test_LF],
    y_test=Y_test_HF,
)
model_LF_HF.model.save(MODEL_DIR / "mfnn_LF_to_HF.keras")

# ----------------------------------------------------
# Train LF + MF → HF
# ----------------------------------------------------
logging.info("Training model: LF + MF → HF")

model_LF_MF_HF = MultiFidelityNN(
    input_shapes=[(2,), (n_points,), (n_points,)],
    coeff=1e-8,
    layers_config=layers_config_2,
    train_config=train_config,
    output_units=n_points,
    output_activation="linear",
    merge_mode="concat",
)

model_LF_MF_HF.train(
    X_train_fidelities=[X_train, Y_train_LF, Y_train_MF],
    y_train=Y_train_HF,
    X_test_fidelities=[X_test, Y_test_LF, Y_test_MF],
    y_test=Y_test_HF,
)
model_LF_MF_HF.model.save(MODEL_DIR / "mfnn_LF_MF_to_HF.keras")

# ----------------------------------------------------
# Evaluation and CMAME-ready Plots
# ----------------------------------------------------
logging.info("Generating evaluation figures...")

Y_pred_LF = model_LF_HF.model.predict([X_test, Y_test_LF])
Y_pred_LF_MF = model_LF_MF_HF.model.predict([X_test, Y_test_LF, Y_test_MF])

def rms(a, b): return np.sqrt(np.mean((a - b)**2, axis=1))
err_LF = rms(Y_test_LF, Y_test_HF)
err_MF = rms(Y_test_MF, Y_test_HF)
err_LF_HF = rms(Y_pred_LF, Y_test_HF)
err_LF_MF_HF = rms(Y_pred_LF_MF, Y_test_HF)

# --- Scatter of RMS errors ---
plt.figure(figsize=(5, 4))
plt.scatter(err_LF, err_LF_HF, s=15, c='tab:blue', alpha=0.6, label="LF→HF NN")
plt.scatter(err_MF, err_LF_MF_HF, s=15, c='tab:red', alpha=0.6, label="LF+MF→HF NN")
lims = [0, max(err_LF.max(), err_MF.max())]
plt.plot(lims, lims, 'k--', lw=1)
plt.xlabel("Solver RMS error (vs HF)")
plt.ylabel("NN RMS error (vs HF)")
plt.legend(frameon=False)
plt.title("Profile-wise Reconstruction Accuracy")
plt.tight_layout()
plt.savefig(MODEL_DIR / "RMS_scatter_CMAME.pdf", dpi=600)

# --- Representative profile reconstruction ---
case_id = np.argmax(err_MF)
y_grid = np.linspace(0, 1, n_points)
plt.figure(figsize=(5.6, 2.5))
plt.plot(y_grid, Y_test_HF[case_id], "k-", lw=1.5, label="HF (ref)")
plt.plot(y_grid, Y_test_LF[case_id], "--", lw=1.0, label="LF solver")
plt.plot(y_grid, Y_test_MF[case_id], "--", lw=1.0, label="MF solver")
plt.plot(y_grid, Y_pred_LF[case_id], lw=1.2, label="LF→HF NN")
plt.plot(y_grid, Y_pred_LF_MF[case_id], lw=1.2, label="LF+MF→HF NN")
plt.xlabel(r"$y$")
plt.ylabel(r"$u(y)$")
plt.legend(frameon=False, ncol=2)
plt.title("Outlet Velocity Reconstruction (Representative Case)")
plt.tight_layout()
plt.savefig(MODEL_DIR / "Outlet_profile_CMAME.pdf", dpi=600)
plt.show()

logging.info("✅ Evaluation complete. Figures saved in models/bfs_multifidelity/")
