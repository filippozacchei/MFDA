"""
Identification routine for BFS multi-fidelity case
Schemes: MH / MLDA / MFDA
MFDA uses two NN levels:
  • Level 1: LF → NN(HF)
  • Level 2: LF + MF → NN(HF)
"""

import os, sys, timeit, numpy as np, arviz as az
from scipy.optimize import least_squares
import tensorflow as tf
import tinyDA as tda
from tensorflow.keras.models import load_model

# -------------------------------------------------------------------------
# Reproducibility setup
# -------------------------------------------------------------------------
import random
import numpy as np
import tensorflow as tf

SEED = 4892  # or any integer of your choice
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"  # ensure deterministic TensorFlow ops

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
MODEL_DIR = "models/bfs_multifidelity"
noise = 0.01
n_iter = 120
burnin = 20
thin = 1
n_chains = 1
sub_sampling = [5, 5]
case = "MFDA"   # Options: "MH", "MLDA", "MFDA"

# -------------------------------------------------------------------------
# Load solvers
# -------------------------------------------------------------------------

from runner_lf import run_lf_case as solver_lf
from runner_mf import run_mf_case as solver_mf
from runner_hf import run_hf_case as solver_hf


# -------------------------------------------------------------------------
# Load pre-trained Multi-Fidelity NNs
# -------------------------------------------------------------------------
model_LF_HF    = load_model(os.path.join(MODEL_DIR, "mfnn_LF_to_HF.keras"), safe_mode=False)
model_LF_MF_HF = load_model(os.path.join(MODEL_DIR, "mfnn_LF_MF_to_HF.keras"), safe_mode=False)

@tf.function(jit_compile=True)
def nn_LF_HF(params, y_lf):
    return model_LF_HF([params, y_lf], training=False)[0]

@tf.function(jit_compile=True)
def nn_LF_MF_HF(params, y_lf, y_mf):
    return model_LF_MF_HF([params, y_lf, y_mf], training=False)[0]

# -------------------------------------------------------------------------
# Prior and truth
# -------------------------------------------------------------------------
class CustomUniform:
    def __init__(self, lower, upper):
        self.lower, self.upper = np.array(lower), np.array(upper)
        self.area = np.prod(self.upper - self.lower)
    def logpdf(self, x):
        return -np.inf if np.any((x < self.lower) | (x > self.upper)) else -np.log(self.area)
    def rvs(self):
        return np.random.uniform(self.lower, self.upper)

x_prior = CustomUniform([0.06, 0.52], [0.14, 1.48])
x_true = x_prior.rvs()
print("True parameters:", x_true)

# -------------------------------------------------------------------------
# Forward models
# -------------------------------------------------------------------------
def model_LF(x):
    y, u = solver_lf(H_in=x[0], U_in=x[1])
    if isinstance(u, float) or np.any(np.isnan(u)):
        return np.zeros(100)  # fallback vector, same dimension as profiles
    return u.flatten()

def model_MF(x):
    y, u = solver_mf(H_in=x[0], U_in=x[1])
    if isinstance(u, float) or np.any(np.isnan(u)):
        return np.zeros(100)
    return u.flatten()

def model_HF(x):
    y, u = solver_hf(H_in=x[0], U_in=x[1])
    if isinstance(u, float) or np.any(np.isnan(u)):
        return np.zeros(100)
    return u.flatten()

def model_NN_LF_HF(x):
    """NN prediction using LF only."""
    y_lf = model_LF(x)[np.newaxis, :]
    x_in = x[np.newaxis, :]
    return nn_LF_HF(x_in, y_lf).numpy().flatten()

def model_NN_LF_MF_HF(x):
    """NN prediction using LF + MF."""
    y_lf = model_LF(x)[np.newaxis, :]
    y_mf = model_MF(x)[np.newaxis, :]
    x_in = x[np.newaxis, :]
    return nn_LF_MF_HF(x_in, y_lf, y_mf).numpy().flatten()

# -------------------------------------------------------------------------
# Synthetic observation (HF reference)
# -------------------------------------------------------------------------
y_true = model_HF(x_true)
y_obs = y_true + np.random.normal(scale=noise, size=y_true.shape)

# -------------------------------------------------------------------------
# Least-squares initialization
# -------------------------------------------------------------------------
def residual(x): return y_obs - model_NN_LF_HF(x)
res = least_squares(residual, x_true*0.9,
                    bounds=([0.05, 0.5], [0.15, 1.5]), jac="3-point")
cov_p = np.linalg.inv(res.jac.T @ res.jac)
cov_p *= 1 / np.max(np.abs(cov_p))
print("Initial guess:", res.x)
print(cov_p)

# -------------------------------------------------------------------------
# Likelihoods and posterior setup
# -------------------------------------------------------------------------
cov_like = np.diag(np.full(y_obs.shape[0], noise**2))

if case == "MH":
    like_hf = tda.GaussianLogLike(y_obs, cov_like)
    posteriors = [tda.Posterior(x_prior, like_hf, model_HF)]
    level = 1

elif case == "MLDA":
    like_lf = tda.AdaptiveGaussianLogLike(y_obs, cov_like)
    like_mf = tda.AdaptiveGaussianLogLike(y_obs, cov_like)
    like_hf = tda.GaussianLogLike(y_obs, cov_like)
    posteriors = [
        tda.Posterior(x_prior, like_lf, model_LF),
        tda.Posterior(x_prior, like_mf, model_MF),
        tda.Posterior(x_prior, like_hf, model_HF)
    ]
    level = 2

elif case == "MFDA":
    # Level 1: LF → NN(HF)
    like_nn1 = tda.AdaptiveGaussianLogLike(y_obs, cov_like)
    # Level 2: LF + MF → NN(HF)
    like_nn2 = tda.GaussianLogLike(y_obs, cov_like)
    posteriors = [
        tda.Posterior(x_prior, like_nn1, model_NN_LF_HF),
        tda.Posterior(x_prior, like_nn2, model_NN_LF_MF_HF)
    ]
    level = 'fine'
    sub_sampling = 10

# -------------------------------------------------------------------------
# MCMC sampling
# -------------------------------------------------------------------------
proposal = tda.GaussianRandomWalk(C=cov_p, scaling=1e-3,
                                  adaptive=True, gamma=1.1, period=10)

start = timeit.default_timer()
samples = tda.sample(posteriors, 
                     proposal,
                     iterations=n_iter, 
                     adaptive_error_model='state-independent',
                     n_chains=n_chains, 
                     initial_parameters=x_true,        
                     force_sequential=True,
                     subchain_length=sub_sampling, 
                     store_coarse_chain=False)
elapsed = timeit.default_timer() - start

# -------------------------------------------------------------------------
# Results
# -------------------------------------------------------------------------
idata = tda.to_inference_data(samples, level=level).sel(
    draw=slice(burnin, None, thin), groups="posterior"
)
idata.to_netcdf(f"samples_{case}.nc")

ess = az.ess(idata)
mean_ess = np.mean([ess.data_vars[f"x{j}"].values for j in range(2)])
val = idata.posterior.mean().to_array()
err = np.mean(np.sqrt((x_true - val) ** 2))

print(f"[{case}] Time: {elapsed:.2f}s | ESS: {mean_ess:.2f} | Time/ESS: {elapsed/mean_ess:.2f} | Err: {err:.3e}")

# -------------------------------------------------------------------------
# Posterior diagnostics and visualisation
# -------------------------------------------------------------------------
import matplotlib.pyplot as plt
import corner

# Extract posterior samples
samples_arr = np.column_stack([
    idata.posterior["x0"].values.flatten(),
    idata.posterior["x1"].values.flatten()
])
param_names = [r"$H_{in}$", r"$U_{in}$"]

# -----------------------------
# 1) Corner plot (joint posterior)
# -----------------------------
fig = corner.corner(
    samples_arr,
    labels=param_names,
    truths=x_true,
    color="C0",
    truth_color="k",
    smooth=1.0,
    levels=(0.5, 0.9),
    show_titles=True,
    title_fmt=".3f",
)
fig.suptitle(f"{case}: Posterior distribution", fontsize=12, y=1.02)
fig.savefig(f"posterior_corner_{case}.pdf", bbox_inches="tight", dpi=600)

# -----------------------------
# 2) Marginal distributions
# -----------------------------
fig, axs = plt.subplots(1, 2, figsize=(6, 2.5))
for j, name in enumerate(param_names):
    axs[j].hist(samples_arr[:, j], bins=30, density=True, color="C0", alpha=0.7)
    axs[j].axvline(x_true[j], color="k", lw=1.2, label="True")
    axs[j].set_xlabel(name)
    axs[j].set_ylabel("pdf")
    axs[j].legend(frameon=False)
plt.tight_layout()
plt.savefig(f"posterior_marginals_{case}.pdf", dpi=600)

# -----------------------------
# 3) Reconstructed vs observed outlet profiles
# -----------------------------
x_post = val.values.squeeze()   # posterior mean parameters
y_pred = model_NN_LF_HF(x_post) if case != "MFDA" else model_NN_LF_MF_HF(x_post)

plt.figure(figsize=(5.5, 2.8))
plt.plot(y_true, "k-", lw=1.5, label="True HF")
plt.plot(y_obs, "C1o", ms=3, label="Noisy obs")
plt.plot(y_pred, "C0--", lw=1.2, label="Posterior mean pred")
plt.xlabel(r"Outlet sample index $i$")
plt.ylabel(r"$u(y_i)$")
plt.legend(frameon=False, ncol=3)
plt.title(f"{case}: Posterior predictive vs observation")
plt.tight_layout()
plt.savefig(f"posterior_profile_{case}.pdf", dpi=600)
plt.show()

# -----------------------------
# 4) Trace plots for convergence (ArviZ)
# -----------------------------
az.plot_trace(idata, var_names=["x0", "x1"])
plt.tight_layout()
plt.savefig(f"traceplots_{case}.pdf", dpi=600)
plt.show()

print("\n✅ Plots saved:")
print(f"  posterior_corner_{case}.pdf")
print(f"  posterior_marginals_{case}.pdf")
print(f"  posterior_profile_{case}.pdf")
print(f"  traceplots_{case}.pdf")
