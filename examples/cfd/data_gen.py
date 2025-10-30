"""
Minimal multi-fidelity dataset generator for BFS flow
=====================================================

Generates outlet velocity profiles from low-, medium-, and high-
fidelity solvers for use in multi-fidelity neural network training.

Outputs:
  data_bfs_multifidelity/params.csv  - sampled (H_in, U_in)
  data_bfs_multifidelity/dataset.csv - outlet velocity data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import qmc

from runner_lf import run_lf_case
from runner_mf import run_mf_case
from runner_hf import run_hf_case

# ----------------------------
# Setup
# ----------------------------
OUT_DIR = Path("data_bfs_multifidelity")
OUT_DIR.mkdir(exist_ok=True)

H_range = (0.05, 0.15)
U_range = (0.5, 1.5)
N_samples = 150
N_points = 100

# ----------------------------
# Latin Hypercube sampling
# ----------------------------
sampler = qmc.LatinHypercube(d=2)
lhs = qmc.scale(sampler.random(N_samples),
                [H_range[0], U_range[0]],
                [H_range[1], U_range[1]])
H_in, U_in = lhs[:, 0], lhs[:, 1]
pd.DataFrame({"H_in": H_in, "U_in": U_in}).to_csv(OUT_DIR / "params.csv", index=False)

# ----------------------------
# Helper
# ----------------------------
def run_solver(solver, label):
    data = []
    for H, U in tqdm(zip(H_in, U_in), total=N_samples, desc=f"{label}"):
        try:
            y, u = solver(H_in=H, U_in=U)
            y_uni = np.linspace(min(y), max(y), N_points)
            u_uni = np.interp(y_uni, y, u)
            data.append(pd.DataFrame({
                "H_in": H, "U_in": U,
                "y": y_uni, "u": u_uni,
                "fidelity": label
            }))
        except Exception as e:
            print(f"[{label}] Failed: H={H:.3f}, U={U:.3f} -> {e}")
    return pd.concat(data, ignore_index=True)

# ----------------------------
# Run all fidelities
# ----------------------------
df = pd.concat([
    run_solver(run_lf_case, "LF"),
    run_solver(run_mf_case, "MF"),
    run_solver(run_hf_case, "HF")
], ignore_index=True)

df.to_csv(OUT_DIR / "dataset.csv", index=False)
print(f"\n✅ Dataset saved to {OUT_DIR/'dataset.csv'} ({len(df):,} rows)")
