"""
Compare results from low-, medium-, and high-fidelity solvers
=============================================================

Loads the generated BFS dataset (dataset.csv) and:
 - Computes RMS error between fidelities
 - Plots representative outlet profiles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Load dataset
# ----------------------------
DATA_PATH = Path("data_bfs_multifidelity/dataset.csv")
df = pd.read_csv(DATA_PATH)
assert not df.empty, "Dataset is empty. Run the generator first."
print(f"Loaded {len(df):,} rows from {DATA_PATH}")

# ----------------------------
# Group into outlet profiles
# ----------------------------
def to_profiles(subdf):
    grouped = subdf.groupby(["H_in", "U_in"])
    X, Y = [], []
    for (H, U), g in grouped:
        g = g.sort_values("y")
        X.append([H, U])
        Y.append(g["u"].values)
    return np.array(X), np.array(Y)

fmap = {"LF": 0, "MF": 1, "HF": 2}
df["fid"] = df["fidelity"].map(fmap)
X_LF, Y_LF = to_profiles(df[df.fid == 0])
X_MF, Y_MF = to_profiles(df[df.fid == 1])
X_HF, Y_HF = to_profiles(df[df.fid == 2])

# ----------------------------
# Align by nearest parameters (tolerant merge)
# ----------------------------
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
    return Y_ref, np.array(Y_tar_aligned)

Y_HF_al, Y_LF_al = align_nearest(X_HF, Y_HF, X_LF, Y_LF)
_, Y_MF_al = align_nearest(X_HF, Y_HF, X_MF, Y_MF)

# Drop invalid matches
mask = ~np.isnan(Y_LF_al).any(axis=1) & ~np.isnan(Y_MF_al).any(axis=1)
Y_HF_al, Y_LF_al, Y_MF_al = Y_HF_al[mask], Y_LF_al[mask], Y_MF_al[mask]
print(f"Aligned {len(Y_HF_al)} consistent samples across fidelities.")

# ----------------------------
# RMS error statistics
# ----------------------------
def rms(a, b): return np.sqrt(np.mean((a - b)**2, axis=1))
err_LF = rms(Y_LF_al, Y_HF_al)
err_MF = rms(Y_MF_al, Y_HF_al)
print(f"\nMean RMS Error vs HF:")
print(f"  LF solver: {err_LF.mean():.4e}")
print(f"  MF solver: {err_MF.mean():.4e}")

# ----------------------------
# Plot representative profiles
# ----------------------------
y = np.linspace(0, 1, Y_HF_al.shape[1])
case_id = np.argmin(err_LF)  # choose a typical, not identical, case

plt.figure(figsize=(5.8, 3))
plt.plot(y, Y_HF_al[case_id], "k-", lw=1.6, label="HF (ref)")
plt.plot(y, Y_MF_al[case_id], "r--", lw=1.3, label="MF solver")
plt.plot(y, Y_LF_al[case_id], "b-.", lw=1.3, label="LF solver")
plt.xlabel(r"$y$")
plt.ylabel(r"$u(y)$")
plt.title("Representative Outlet Velocity Profiles")
plt.legend(frameon=False, ncol=3)
plt.tight_layout()
plt.savefig("solver_comparison_CMAME.pdf", dpi=600)
plt.show()
