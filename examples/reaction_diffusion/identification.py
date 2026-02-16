# Standard Library Imports
import os
import time

import sys
import timeit
from scipy.optimize import least_squares
from timeit import repeat

# Third-party Library Imports
import numpy as np
import pandas as pd
import tensorflow as tf
import arviz as az
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, multivariate_normal, uniform
from tensorflow.keras.models import load_model
from itertools import product
from numba import jit
import json
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
os.environ['OMP_NUM_THREADS'] = "48"

# Local Module Imports
sys.path.append('./data/')
sys.path.append('./models/')
sys.path.append('../src/forward_models')
import tinyDA as tda

from model import Model_DR
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.extend([os.path.join(BASE_DIR, 'forward_models'), os.path.join(BASE_DIR, 'utils')])
from data_utils import load_hdf5, prepare_lstm_dataset
from pod_utils import reshape_to_pod_2d_system_snapshots, project, reshape_to_lstm, reconstruct, reconstruct_eff, reconstruct_eff1

# Import necessary modules
from multi_fidelity_lstm import MultiFidelityLSTM

start_time = timeit.default_timer()

case = "FOM"  #
print(case)

# MCMC Parameters
sample_id    = 1
noise        = 0.2
scaling      = 0.001
scaling1     = 1
scaling2     = 1
scaling3     = 1
n_iter       = 1000
burnin       = 0
thin         = 1
num_modes    = 25
sub_sampling = [5,5]


def load_configuration(config_path):
    """Load JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def temporal_interpolation_splines2(u_data_coarse, time_steps_coarse, time_steps_fine):
    """
    Perform temporal interpolation using cubic splines on u_data_coarse
    to match the time resolution of high-fidelity data.
    More efficient by reducing nested loops via reshaping.
    """
    num_samples, _, n, n = u_data_coarse.shape

    # Compute index mapping from fine time to coarse time
    coarse_idx = np.floor(
        np.linspace(0, time_steps_coarse - 1, time_steps_fine)
    ).astype(int)

    u_data_interpolated = u_data_coarse[:, coarse_idx, :, :]

    return u_data_interpolated

def load_and_process_data(config, num_modes=25):
    """
    Load datasets, apply POD projection, and prepare LSTM-ready data.
    :param config: Configuration dictionary.
    :param num_modes: Number of POD modes to retain.
    :return: Processed training and testing data.
    """
    train_data = load_hdf5(config["train"])
    train_data_coarse3 = load_hdf5(config["train_coarse3"])
    pod_basis_coarse3 = load_hdf5(config["pod_basis_coarse3"])
    
    print(train_data_coarse3['u'].shape) 
    train_data_coarse3['u'] = temporal_interpolation_splines2(
        train_data_coarse3['u'], train_data_coarse3['u'].shape[1], train_data['u'].shape[1]
    )
    
    
    train_data_coarse3['v'] = temporal_interpolation_splines2(
        train_data_coarse3['v'], train_data_coarse3['v'].shape[1], train_data['v'].shape[1]
    )
    
    u_train_snapshots_coarse3 = reshape_to_pod_2d_system_snapshots(train_data_coarse3['u'], train_data_coarse3['v'])

    U_coarse3, Sigma_coarse3 = pod_basis_coarse3["POD_modes_coarse"], pod_basis_coarse3["singular_values_fine"]
    v_train_coarse3 = project(U_coarse3, Sigma_coarse3, u_train_snapshots_coarse3, num_modes=num_modes)
    print(train_data['u'].shape[0])
    print(train_data['u'].shape[1])
    v_train_lstm_coarse3 = reshape_to_lstm(v_train_coarse3, train_data['u'].shape[0], train_data['u'].shape[1], num_modes)

    # Prepare additional input features
    X_train_init = np.column_stack((train_data["d1"], train_data["beta"]))

    X_train_prep_coarse3 = prepare_lstm_dataset(X_train_init, train_data["t"], v_train_lstm_coarse3)
    
    # Split into features and targets
    X_train = X_train_prep_coarse3[:, :, :3]
    
    scaler_X = StandardScaler()

    X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    
    return scaler_X

# Initialize Parameters
n_samples = 25
np.random.seed(123)
config_filepath = 'config/config_MultiFidelity_3.json'
config = load_configuration(config_filepath)
scaler_X = load_and_process_data(config, num_modes=num_modes)

# Instantiate Models for Different Resolutions
solver_h1 = Model_DR(n=128, dt=0.2, L=20., T=50.05)
solver_h2 = Model_DR(n=64,  dt=0.2, L=20., T=50.05)
solver_h3 = Model_DR(n=32,  dt=0.5, L=20., T=50.05)
solver_h4 = Model_DR(n=16,  dt=1.0, L=20., T=50.05)

# Define Points for Data Extraction1

x_data = y_data = np.array([-7.5,-6.25,-5.0,-3.75,-2.5,-1.25,0.0,1.25,2.5,3.75,5.0,6.25,7.5])
datapoints = np.array(list(product(x_data, y_data)))
i_indices = np.array([np.argmin(np.abs(solver_h1.x - x)) for x in datapoints[:, 0]])
j_indices = np.array([np.argmin(np.abs(solver_h1.y - y)) for y in datapoints[:, 1]])

def load_configuration(config_path):
    """Load JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)
    
def temporal_interpolation_splines(u_data_coarse, time_steps_coarse, time_steps_fine):
    n, _, _ = u_data_coarse.shape
    time_coarse = np.linspace(0, 1, time_steps_coarse)
    time_fine = np.linspace(0, 1, time_steps_fine)

    u_data_flat = u_data_coarse.reshape(n * n, -1)
    splines = interp1d(time_coarse, u_data_flat, kind='cubic', axis=1, fill_value="extrapolate")
    interpolated = splines(time_fine).reshape(n, n, time_steps_fine).transpose(2, 0, 1)
    return interpolated[np.newaxis]

# Solver Data Functions
def solver_data(solver, x, lag=1):
    """
    Run the solver and extract data for given coordinates.

    Parameters:
    solver : Model_DR instance
        The solver object to run.
    x : array-like
        Input parameter for the solver.

    Returns:
    np.ndarray
        Array of (u, v) data for specified points.
    """
    # Run the solver once with the given parameters
    solver.solve(x[0], x[1])

    # Use the updated get_data method to retrieve data in one call
    return solver.get_data(datapoints, lag)

def get_data(sol_u, sol_v, lag=1):
    data = np.stack([sol_u[i_indices, j_indices, ::lag], sol_v[i_indices, j_indices, ::lag]], axis=1)
    return data
    
def model_HF(input):  return solver_data(solver_h1, input).flatten()
def model_LF1(input): return solver_data(solver_h2, input).flatten()
def model_LF2(input): return solver_data(solver_h3, input).flatten()
def model_LF3(input): return solver_data(solver_h4, input).flatten()

def model_HF_lag(input):  return solver_data(solver_h1, input, lag=5).flatten()
def model_LF1_lag(input): return solver_data(solver_h2, input, lag=5).flatten()
def model_LF2_lag(input): return solver_data(solver_h3, input, lag=2).flatten()
def model_LF3_lag(input): return solver_data(solver_h4, input, lag=1).flatten()

d1_range = [0.01,0.1]
beta_range = [0.5,1.5]

d1_scale = d1_range[1] - d1_range[0]
beta_scale = beta_range[1] - beta_range[0]

class CustomUniform:
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.area = (self.upper_bound[0] - self.lower_bound[0])*(self.upper_bound[1] - self.lower_bound[1])
    
    def pdf(self, x):
        if (self.lower_bound[0] <= x[0] <= self.upper_bound[0]) and (self.lower_bound[1] <= x[1] <= self.upper_bound[1]):
            return 1 / self.area
        else:
            return 0
        
    def logpdf(self, x):   
        if self.pdf(x) == 0:
            return -np.inf
        else:
            return np.log(self.pdf(x))
    
    def rvs(self):
        return np.random.uniform(self.lower_bound, self.upper_bound)

def interpolate_and_reshape(data, time_steps_coarse, time_steps_fine, grid_points):
    interpolated_data = temporal_interpolation_splines(data, time_steps_coarse, time_steps_fine)
    return interpolated_data.reshape(time_steps_fine, grid_points).T

def expand_parameters(input, time_steps_fine):
    input = input.flatten()[np.newaxis, np.newaxis, :]  # Ensure input is 3D
    return np.repeat(input, time_steps_fine, axis=1)  # Repeat across the time dimension

x_distribution = CustomUniform([0.01,0.5],[0.1,1.5])

if case == "MFDA1":
    # Load Models for Low- and Multi-fidelity Predictions
    models_1 = load_model(f'models/multi_fidelity_bis/resolution_h4/model.keras', safe_mode=False)
    models_2 = load_model(f'models/multi_fidelity_bis/resolution_h3-h4/model.keras', safe_mode=False)
    models_3 = load_model(f'models/multi_fidelity_bis/resolution_h2-h3-h4/model.keras', safe_mode=False)
    config_filepath = 'config/config_MultiFidelity_3.json'
    config = load_configuration(config_filepath)
    pod_basis = load_hdf5(config["pod_basis"])
    pod_basis_coarse1 = load_hdf5(config["pod_basis_coarse1"])
    pod_basis_coarse2 = load_hdf5(config["pod_basis_coarse2"])
    pod_basis_coarse3 = load_hdf5(config["pod_basis_coarse3"])
    
    @tf.function(jit_compile=True)
    def model_mf1(input1, input2):
        return models_1([input1,input2], training=False)[0]
    
    @tf.function(jit_compile=True)
    def model_mf2(input1, input2, input3):
        return models_2([input1,input2,input3], training=False)[0]
    
    @tf.function(jit_compile=True)
    def model_mf3(input1, input2, input3, input4):
        return models_3([input1,input2,input3,input4], training=False)[0]
    
    grid_points_h4 = 16*16
    grid_points_h3 = 32*32
    grid_points_h2 = 64*64
    grid_points_h1 = 128*128
    time_steps_coarseh4 = 51
    time_steps_coarseh3 = 101
    time_steps_coarseh2 = 251
    time_steps_fine = 251
    time_fine = np.arange(0, 50.05, 0.2)
    time_expanded = np.tile(time_fine, (1, 1))[:, :, np.newaxis]  # (num_samples, time_steps, 1)
    
    U, Sigma = pod_basis["POD_modes"], pod_basis["singular_values"]
    U_coarse1, Sigma_coarse1 = pod_basis_coarse1["POD_modes_coarse"], pod_basis_coarse1["singular_values_fine"]
    U_coarse2, Sigma_coarse2 = pod_basis_coarse2["POD_modes_coarse"], pod_basis_coarse2["singular_values_fine"]
    U_coarse3, Sigma_coarse3 = pod_basis_coarse3["POD_modes_coarse"], pod_basis_coarse3["singular_values_fine"]

    def prepare_coarse_data(solver, U_coarse, Sigma_coarse, input_data, time_steps_coarse, grid_points, num_modes):
        coarse_u, coarse_v = solver.solve(input_data[0], input_data[1])
        coarse_u = interpolate_and_reshape(coarse_u, time_steps_coarse, time_steps_fine, grid_points)
        coarse_v = interpolate_and_reshape(coarse_v, time_steps_coarse, time_steps_fine, grid_points)
        
        coarse_data = np.concatenate((coarse_u, coarse_v), axis=0)
        coarse_data = project(U_coarse, Sigma_coarse, coarse_data, num_modes=num_modes)
        coarse_data = coarse_data.reshape(1, time_steps_fine, num_modes)
        
        return coarse_data

    def prepare_input(input_data, time_steps_fine):
        param_expanded = expand_parameters(input_data, time_steps_fine)
        X = np.concatenate([param_expanded, time_expanded], axis=2)
        return X

    def model_1(input_data):
        coarse_data = prepare_coarse_data(solver_h4, U_coarse3, Sigma_coarse3, input_data, time_steps_coarseh4, grid_points_h4, num_modes)
        X = prepare_input(input_data, time_steps_fine)
        X = scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        predictions = model_mf1(X, coarse_data * Sigma[:num_modes]).numpy()
        sol_u, sol_v = reconstruct_eff1(U,Sigma,predictions/Sigma[:num_modes],25)
        sol = get_data(sol_u, sol_v, lag=5).flatten()
        return sol

    def model_2(input_data):
        coarse_data1 = prepare_coarse_data(solver_h4, U_coarse3, Sigma_coarse3, input_data, time_steps_coarseh4, grid_points_h4, num_modes)
        coarse_data2 = prepare_coarse_data(solver_h3, U_coarse2, Sigma_coarse2, input_data, time_steps_coarseh3, grid_points_h3, num_modes)
        X = prepare_input(input_data, time_steps_fine)
        X = scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        predictions = model_mf2(X, coarse_data2 * Sigma[:num_modes], coarse_data1 * Sigma[:num_modes]).numpy()
        sol_u, sol_v = reconstruct_eff1(U,Sigma,predictions/Sigma[:num_modes],25)
        sol = get_data(sol_u, sol_v, lag=5).flatten()
        return sol

    def model_3(input_data):
        coarse_data1 = prepare_coarse_data(solver_h4, U_coarse3, Sigma_coarse3, input_data, time_steps_coarseh4, grid_points_h4, num_modes)
        coarse_data2 = prepare_coarse_data(solver_h3, U_coarse2, Sigma_coarse2, input_data, time_steps_coarseh3, grid_points_h3, num_modes)
        coarse_data3 = prepare_coarse_data(solver_h2, U_coarse1, Sigma_coarse1, input_data, time_steps_coarseh2, grid_points_h2, num_modes)
        X = prepare_input(input_data, time_steps_fine)
        X = scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        predictions = model_mf3(X, coarse_data3 * Sigma[:num_modes], coarse_data2 * Sigma[:num_modes], coarse_data1 * Sigma[:num_modes]).numpy()
        sol_u, sol_v = reconstruct_eff1(U,Sigma,predictions/Sigma[:num_modes],25)
        sol = get_data(sol_u, sol_v, lag=5).flatten()
        return sol
        
    input_data = np.array([0.05,0.5])
    
    model_1_output = model_1(input_data)
    model_2_output = model_2(input_data)
    model_3_output = model_3(input_data)
    
    # ==============================================
    # Forward Model Cost & Accuracy for All Levels
    # ==============================================

    def rmse(a, b):
        return np.sqrt(np.mean((a - b)**2))

    models = {
        "HF"    : model_HF_lag,
        "LF(1)" : model_LF3_lag,
        "LF(2)" : model_LF2_lag,
        "LF(3)" : model_LF1_lag,
        "MFDA(1)" : model_1,
        "MFDA(2)" : model_2,
        "MFDA(3)" : model_3,
    }

    # Select some representative test parameters
    np.random.seed(123)
    theta_tests = [x_distribution.rvs() for _ in range(20)]

    """
    records = []

    print("\nRunning forward accuracy/time evaluation...\n")

    for theta in theta_tests:
        # compute high-fidelity once
        t0 = time.time()
        y_ref = models["HF"](theta)
        t_ref = time.time() - t0

        for name, fwd in models.items():
            t0 = time.time()
            y_pred = fwd(theta)
            t_eval = time.time() - t0
            err = rmse(y_pred, y_ref)
            records.append([name, theta[0], theta[1], t_eval, err])

    df_raw = pd.DataFrame(records, columns=["Model", "d1", "beta", "Time (s)", "RMSE vs HF"])

    # Compute mean ± std per model
    df_summary = df_raw.groupby("Model").agg({
        "Time (s)"     : ['mean','std'],
        "RMSE vs HF"   : ['mean','std']
    }).reset_index()

    # Formatting for readability
    df_summary.columns = ["Model", "Time Mean (s)", "Time Std (s)", "RMSE Mean", "RMSE Std"]
    print("\n=== Performance Summary (20 random samples) ===\n")
    print(df_summary.to_string(index=False))

    print("\n=== Raw measurements (all samples) ===\n")
    print(df_raw.head())
    """
elif case == "MLDA1":
    
    model_1 = model_LF3_lag
    model_2 = model_LF2_lag
    model_3 = model_LF1_lag
    model_4 = model_HF_lag

Times, Time_ESS, ESS, samples_tot, ERR = [], [], [], [], []

for i in range(sample_id):
    x_true = x_distribution.rvs()   
print(x_true, flush=True)

y_true = model_HF(x_true)
print(y_true.shape, flush=True)

y_hf  = solver_h1.get_data(datapoints, lag=5).flatten()
y_lf1 = solver_h1.get_data(datapoints, lag=5).flatten()
y_lf2 = solver_h1.get_data(datapoints, lag=5).flatten()
y_lf3 = solver_h1.get_data(datapoints, lag=5).flatten()

y_observed = y_hf + np.random.normal(scale=noise, size=y_hf.shape[0])
y_obs1 = y_lf1 + np.random.normal(scale=noise, size=y_lf1.shape[0])
y_obs2 = y_lf2 + np.random.normal(scale=noise, size=y_lf2.shape[0])
y_obs3 = y_lf3 + np.random.normal(scale=noise, size=y_lf3.shape[0])
    
#def ls(x):
#    return (y_observed-model_HF_lag(x))

#x_initial = x_true
#x_initial[0] = x_initial[0] if x_initial[0] > 0.01 else x_true[0]
#x_initial[1] = x_initial[1] if x_initial[1] > 0.5 else x_true[1]

#res = least_squares(ls,x_initial, jac='3-point', verbose=2, bounds=([0.01,0.5],[0.1,1.5]))
#covariancep = np.linalg.inv(res.jac.T @ res.jac)
#covariancep *= 1/np.max(np.abs(covariancep))
#print("Covariance: ", covariancep)
#print("Params: ", res.x)

n = y_observed.shape[0]
cov_likelihood = np.diag(np.full(n, noise**2,dtype=np.float32))
y_distribution_fine = tda.GaussianLogLike(y_observed, cov_likelihood)

#x_distribution = CustomUniform([0.01,0.5],[0.1,1.5])

import numpy as np
from scipy import stats

# original bounds (per dimension)
a = np.array([0.01, 0.5])
b = np.array([0.10, 1.5])

# match Uniform(a,b) mean and variance with a Normal
mu = (a + b) / 2.0
prior_sigma = (b - a) / np.sqrt(12.0)
cov_prior = np.diag(prior_sigma**2)

# independent 1D normals (one per dimension)
x_distribution = stats.multivariate_normal(mean=mu, cov=cov_prior)
#my_proposal = tda.GaussianRandomWalk(C=covariancep,scaling=1e-3, adaptive=True, gamma=1.1, period=10)
my_proposal = tda.AdaptiveMetropolis(C0=scaling*cov_prior, adaptive=True, gamma=1.01, period=100)
x_initial = mu
if case == "MFDA1":

    y_distribution_1 = tda.GaussianLogLike(y_observed+model_1(x_initial)-model_3(x_initial), cov_likelihood*scaling1)
    y_distribution_2 = tda.GaussianLogLike(y_observed+model_2(x_initial)-model_3(x_initial), cov_likelihood*scaling2)
    y_distribution_3 = tda.GaussianLogLike(y_observed, cov_likelihood*scaling3)

    my_posteriors = [ 
        tda.Posterior(x_distribution, y_distribution_1, model_1),
        tda.Posterior(x_distribution, y_distribution_2, model_2),
        tda.Posterior(x_distribution, y_distribution_3, model_3)
    ]

    level = 2

    sub_sampling = [5,5]
    
elif case == "MLDA1":
    cov_lf1 = noise**2 * np.eye(y_lf1.shape[0])
    cov_lf2 = noise**2 * np.eye(y_lf2.shape[0])
    cov_lf3 = noise**2 * np.eye(y_lf3.shape[0])
    cov_hf  = noise**2 * np.eye(y_hf.shape[0])
    y_distribution_1 = tda.GaussianLogLike(y_lf1 + model_1(x_initial) - model_4(x_initial), cov_lf1*scaling1)
    y_distribution_2 = tda.GaussianLogLike(y_lf2 + model_2(x_initial) - model_4(x_initial), cov_lf2*scaling1)
    y_distribution_3 = tda.GaussianLogLike(y_lf3 + model_3(x_initial) - model_4(x_initial), cov_lf3*scaling2)
    y_distribution_4 = tda.GaussianLogLike(y_hf, cov_hf*scaling2)

    my_posteriors = [
        tda.Posterior(x_distribution, y_distribution_1, model_1), 
        tda.Posterior(x_distribution, y_distribution_2, model_2), 
        tda.Posterior(x_distribution, y_distribution_3, model_3),
        tda.Posterior(x_distribution, y_distribution_4, model_4)
    ] 

    level = 3

    sub_sampling = [5,5,1]

else:
    my_posteriors= tda.Posterior(x_distribution, y_distribution_fine, model_HF_lag)
    level = 2

samples = tda.sample(my_posteriors, my_proposal, iterations=n_iter, n_chains=5,
                    initial_parameters=x_initial, subchain_length=sub_sampling,store_coarse_chain=False,force_sequential=True)
elapsed_time = timeit.default_timer() - start_time
idata = tda.to_inference_data(samples, level=level).sel(draw=slice(burnin, None, thin), groups="posterior")
idata.to_netcdf("samples_" + case + "_" + str(sample_id) + ".nc")
ess = az.ess(idata)
rhat = az.rhat(idata)
variables = [v for v in ess.data_vars if v.startswith('x')]

mean_ess = np.mean([ess.data_vars[f'x{j}'].values for j in range(2)])
mean_rhat = np.mean([rhat.data_vars[v].values.flatten() for v in variables])
    
n_samples = 10

# Store Results
Times.append(elapsed_time)
ESS.append(mean_ess)
Time_ESS.append(elapsed_time / mean_ess)
post = idata.posterior
val=post.mean().to_array()
err=(np.mean(np.sqrt((x_true-val)**2)))
ERR.append(err)
print(f'Time: {elapsed_time:.2f}, ESS: {mean_ess:.2f}, Time/ESS: {elapsed_time / mean_ess:.2f}, Err: {err:.6f} ({i}/{n_samples})')

posterior = idata.posterior

# Extract draws as array: (chains, draws, params)
arr = np.stack([posterior[v].values for v in variables], axis=-1)  # shape: (chains, draws, vars)

draws = arr.shape[1]  # number of posterior draws after burn-in
rhat_threshold = 1.01

# Walk through the chain and find first index where R_hat ≤ 1.01
conv_index = None
for k in range(50, draws, 10):  # check every 100 draws to reduce cost
    partial_idata = idata.sel(draw=slice(0, k))
    rhat_k = az.rhat(partial_idata)
    mean_rhat_k = np.max([rhat_k.data_vars[v].values.flatten() for v in variables])
    if mean_rhat_k <= rhat_threshold:
        conv_index = k
        break

# If never reached threshold, treat full chain as needed
if conv_index is None:
    conv_index = draws

# Convert convergence index back to iterations (before burnin thinning)
iter_per_draw = thin
conv_iters = conv_index * iter_per_draw

# Adjust runtime proportionally
time_adjusted = elapsed_time * (conv_iters / n_iter)

# Store adjusted runtime
Times_Adjusted = []
Times_Adjusted.append(time_adjusted)

print(f"Convergence Iterations: {conv_iters}, Adjusted Time: {time_adjusted:.2f} s")
