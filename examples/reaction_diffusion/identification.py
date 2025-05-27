# Standard Library Imports
import os

os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import sys
import timeit
from scipy.optimize import least_squares

# Optimize performance by setting environment variables
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'
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

# Local Module Imports
sys.path.append('./data/')
sys.path.append('./models/')
sys.path.append('../src/forward_models')
import tinyDA as tda

from model import Model_DR
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.extend([os.path.join(BASE_DIR, 'forward_models'), os.path.join(BASE_DIR, 'utils')])
from data_utils import load_hdf5, prepare_lstm_dataset
from pod_utils import reshape_to_pod_2d_system_snapshots, project, reshape_to_lstm, reconstruct, reconstruct_eff

# Import necessary modules
from multi_fidelity_lstm import MultiFidelityLSTM

case = "1-step"  # Options: "2-step"/"1-step"/"FOM"
print(case)

# MCMC Parameters
noise        = 0.02
scaling      = 0.05
scaling1     = 1
scaling2     = 1
scaling3     = 1
n_iter       = 2000
burnin       = 0
thin         = 1
num_modes    = 40
sub_sampling = 1

def load_configuration(config_path):
    """Load JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def temporal_interpolation_splines2(u_data_coarse, time_steps_coarse, time_steps_fine):
    """ Perform temporal interpolation on u_data_coarse using cubic splines
    to match the time dimensionality of the high-fidelity data. """
    num_samples, _, n, n = u_data_coarse.shape
    # Create a normalized time grid for coarse and fine data
    time_coarse = np.linspace(0, 1, time_steps_coarse)
    time_fine = np.linspace(0, 1, time_steps_fine)
    
    # Allocate memory for interpolated data
    u_data_coarse_interpolated = np.zeros((num_samples, time_steps_fine, n, n))
    
    # Perform interpolation for each sample and spatial location
    for sample_idx in range(num_samples):
        for i in range(n):
            for j in range(n):
                # Extract the time series for the current spatial location
                time_series = u_data_coarse[sample_idx, :, i, j]
                # Create a cubic spline interpolator
                spline = interp1d(time_coarse, time_series, kind='cubic', fill_value="extrapolate")
                # Interpolate to the fine time grid
                u_data_coarse_interpolated[sample_idx, :, i, j] = spline(time_fine)
    
    return u_data_coarse_interpolated

def load_and_process_data(config, num_modes=40):
    """
    Load datasets, apply POD projection, and prepare LSTM-ready data.
    :param config: Configuration dictionary.
    :param num_modes: Number of POD modes to retain.
    :return: Processed training and testing data.
    """
    train_data = load_hdf5(config["train"])
    train_data_coarse3 = load_hdf5(config["train_coarse3"])
    pod_basis_coarse3 = load_hdf5(config["pod_basis_coarse3"])
    
    
    train_data_coarse3['u'] = temporal_interpolation_splines2(
        train_data_coarse3['u'], train_data_coarse3['u'].shape[1], train_data['u'].shape[1]
    )
    
    
    train_data_coarse3['v'] = temporal_interpolation_splines2(
        train_data_coarse3['v'], train_data_coarse3['v'].shape[1], train_data['v'].shape[1]
    )
    
    u_train_snapshots_coarse3 = reshape_to_pod_2d_system_snapshots(train_data_coarse3['u'], train_data_coarse3['v'])

    U_coarse3, Sigma_coarse3 = pod_basis_coarse3["POD_modes_coarse"], pod_basis_coarse3["singular_values_fine"]
    v_train_coarse3 = project(U_coarse3, Sigma_coarse3, u_train_snapshots_coarse3, num_modes=num_modes)

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
np.random.seed(223)
config_filepath = 'config/config_MultiFidelity_3.json'
config = load_configuration(config_filepath)
scaler_X = load_and_process_data(config, num_modes=num_modes)

# Instantiate Models for Different Resolutions
solver_h1 = Model_DR(n=128, dt=0.05, L=20., T=50.05)
solver_h2 = Model_DR(n=64,  dt=0.1,  L=20., T=50.05)
solver_h3 = Model_DR(n=32,  dt=0.2,  L=20., T=50.05)
solver_h4 = Model_DR(n=16,  dt=0.5,  L=20., T=50.05)

# Define Points for Data Extraction
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

# @jit(nopython=True)
def get_data(sol_u, sol_v, lag=1):
    data = np.stack([sol_u[i_indices, j_indices, ::lag], sol_v[i_indices, j_indices, ::lag]], axis=1)
    return data
    
def model_HF(input):  return solver_data(solver_h1, input).flatten()
def model_LF1(input): return solver_data(solver_h2, input).flatten()
def model_LF2(input): return solver_data(solver_h3, input).flatten()
def model_LF3(input): return solver_data(solver_h4, input).flatten()

def model_HF_lag(input):  return solver_data(solver_h1, input, lag=20).flatten()
def model_LF1_lag(input): return solver_data(solver_h2, input, lag=10).flatten()
def model_LF2_lag(input): return solver_data(solver_h3, input, lag=5).flatten()
def model_LF3_lag(input): return solver_data(solver_h4, input, lag=2).flatten()

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

# @jit(nopython=True)
def interpolate_and_reshape(data, time_steps_coarse, time_steps_fine, grid_points):
    interpolated_data = temporal_interpolation_splines(data, time_steps_coarse, time_steps_fine)
    return interpolated_data.reshape(time_steps_fine, grid_points).T

# @jit(nopython=True)
def expand_parameters(input, time_steps_fine):
    input = input.flatten()[np.newaxis, np.newaxis, :]  # Ensure input is 3D
    return np.repeat(input, time_steps_fine, axis=1)  # Repeat across the time dimension
    
if case == "1-step":
    # Load Models for Low- and Multi-fidelity Predictions
    models_1 = load_model(f'models/multi_fidelity_hyper_3/resolution_h4/model.keras', safe_mode=False)
    models_2 = load_model(f'models/multi_fidelity_hyper_3/resolution_h3-h4/model.keras', safe_mode=False)
    models_3 = load_model(f'models/multi_fidelity_hyper_3/resolution_h2-h3-h4/model.keras', safe_mode=False)
    config_filepath = 'config/config_MultiFidelity_3.json'
    config = load_configuration(config_filepath)
    pod_basis = load_hdf5(config["pod_basis"])
    pod_basis_coarse1 = load_hdf5(config["pod_basis_coarse1"])
    pod_basis_coarse2 = load_hdf5(config["pod_basis_coarse2"])
    pod_basis_coarse3 = load_hdf5(config["pod_basis_coarse3"])
    
    # # Define TensorFlow Functions with JIT Compilation
    # @tf.function(jit_compile=True)
    # def model_lf(input):
    #     input_reshaped = tf.reshape(input, (1, 64))
    #     return tf.reduce_mean([mod(input_reshaped, training=False)[0] for mod in models_l], axis=0)

    #@tf.function(jit_compile=True)
    def model_mf1(input1, input2):
        return models_1([input1,input2], training=False)[0]
    
    #@tf.function(jit_compile=True)
    def model_mf2(input1, input2, input3):
        return models_2([input1,input2,input3], training=False)[0]
    
    #@tf.function(jit_compile=True)
    def model_mf3(input1, input2, input3, input4):
        return models_3([input1,input2,input3,input4], training=False)[0]
    
    # def model_1(input):
    #     coarse_data1 = tf.constant(solver_h4_data(input), dtype=tf.float32)
    #     return model_mf1(input, coarse_data1).numpy().flatten()
    
    grid_points_h4 = 16*16
    grid_points_h3 = 32*32
    grid_points_h2 = 64*64
    grid_points_h1 = 128*128
    time_steps_coarseh4 = 101
    time_steps_coarseh3 = 251
    time_steps_coarseh2 = 501
    time_steps_fine = 1001
    time_fine = np.arange(0, 50.05, 0.05)
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

    # @jit(nopython=True)
    def prepare_input(input_data, time_steps_fine):
        param_expanded = expand_parameters(input_data, time_steps_fine)
        X = np.concatenate([param_expanded, time_expanded], axis=2)
        return X

    def model_1(input_data):
        coarse_data = prepare_coarse_data(solver_h4, U_coarse3, Sigma_coarse3, input_data, time_steps_coarseh4, grid_points_h4, num_modes)
        X = prepare_input(input_data, time_steps_fine)
        X = scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        predictions = model_mf1(X, coarse_data * Sigma[:num_modes]).numpy()
        sol_u, sol_v = reconstruct_eff(U,predictions,40)
        sol = get_data(sol_u, sol_v, lag=20).flatten()
        return sol

    def model_2(input_data):
        coarse_data1 = prepare_coarse_data(solver_h4, U_coarse3, Sigma_coarse3, input_data, time_steps_coarseh4, grid_points_h4, num_modes)
        coarse_data2 = prepare_coarse_data(solver_h3, U_coarse2, Sigma_coarse2, input_data, time_steps_coarseh3, grid_points_h3, num_modes)
        X = prepare_input(input_data, time_steps_fine)
        X = scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        predictions = model_mf2(X, coarse_data2 * Sigma[:num_modes], coarse_data1 * Sigma[:num_modes]).numpy()
        sol_u, sol_v = reconstruct_eff(U,predictions,40)
        sol = get_data(sol_u, sol_v, lag=20).flatten()
        return sol

    def model_3(input_data):
        coarse_data1 = prepare_coarse_data(solver_h4, U_coarse3, Sigma_coarse3, input_data, time_steps_coarseh4, grid_points_h4, num_modes)
        coarse_data2 = prepare_coarse_data(solver_h3, U_coarse2, Sigma_coarse2, input_data, time_steps_coarseh3, grid_points_h3, num_modes)
        coarse_data3 = prepare_coarse_data(solver_h2, U_coarse1, Sigma_coarse1, input_data, time_steps_coarseh2, grid_points_h2, num_modes)
        X = prepare_input(input_data, time_steps_fine)
        X = scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        predictions = model_mf3(X, coarse_data3 * Sigma[:num_modes], coarse_data2 * Sigma[:num_modes], coarse_data1 * Sigma[:num_modes]).numpy()
        sol_u, sol_v = reconstruct_eff(U,predictions,40)
        sol = get_data(sol_u, sol_v, lag=20).flatten()
        return sol
        
    NUM_DATAPOINTS = 2
    input_data = np.array([0.05,0.5])
    
    from timeit import default_timer as timer

    # Measure execution time for model_1
    start_time = timer()
    model_1_output = model_1(input_data)
    execution_time = timer() - start_time
    print("Model MF1: ", execution_time, flush=True)

    # Measure execution time for model_2
    start_time = timer()
    model_2_output = model_2(input_data)
    execution_time = timer() - start_time
    print("Model MF2: ", execution_time, flush=True)

    # Measure execution time for model_3
    # start_time = timer()
    # model_3_output = model_3(input_data)
    # execution_time = timer() - start_time
    # print("Model MF3: ", execution_time, flush=True)

    # Measure execution time for model_LF3
    # start_time = timer()
    # model_LF3_output = model_LF3(input_data)
    # execution_time = timer() - start_time
    # print("Model LF1: ", execution_time, flush=True)

    # # Measure execution time for model_LF2
    # start_time = timer()
    # model_LF2_output = model_LF2(input_data)
    # execution_time = timer() - start_time
    # print("Model LF2: ", execution_time, flush=True)

    # # Measure execution time for model_LF1
    # start_time = timer()
    # model_LF1_output = model_LF1(input_data)
    # execution_time = timer() - start_time
    # print("Model LF3: ", execution_time, flush=True)

    # # Measure execution time for model_HF
    # start_time = timer()
    # model_HF_output = model_HF(input_data)
    # execution_time = timer() - start_time
    # print("Model HF: ", execution_time, flush=True)
    
elif case == "MLDA":
    
    model_1 = model_LF2_lag
    model_2 = model_LF1_lag
    model_3 = model_HF_lag
    

x_distribution = CustomUniform([0.01,0.5],[0.1,1.5])

Times, Time_ESS, ESS, samples_tot, ERR = [], [], [], [], []

x_true = x_distribution.rvs()   
print(x_true, flush=True)

y_true = model_HF(x_true)
print(y_true.shape, flush=True)

y_hf  = solver_h1.get_data(datapoints, lag=20).flatten()
y_lf1 = solver_h1.get_data(datapoints, lag=20).flatten()
y_lf2 = solver_h1.get_data(datapoints, lag=20).flatten()
y_lf3 = solver_h1.get_data(datapoints, lag=20).flatten()


y_observed = y_hf + np.random.normal(scale=noise, size=y_hf.shape[0])
y_obs1 = y_lf1 + np.random.normal(scale=noise, size=y_lf1.shape[0])
y_obs2 = y_lf2 + np.random.normal(scale=noise, size=y_lf2.shape[0])
y_obs3 = y_lf3 + np.random.normal(scale=noise, size=y_lf3.shape[0])
    
def ls(x):
    return (y_observed-model_HF_lag(x))

res = least_squares(ls,[0.08,1.25], jac='3-point', verbose=2, bounds=([0.01,0.5],[0.1,1.5]))
covariancep = np.linalg.inv(res.jac.T @ res.jac)
covariancep *= 1/np.max(np.abs(covariancep))
print("Covariance: ", covariancep)
print("Params: ", res.x)
covariance = np.array([[0.01,0.0],[0.0,0.1]])
# covariancep = np.array([[1.0,0.77],[0.77,0.73]])
print(covariance)
# print(covariancep)

# Likelihood Distributions
n = y_true.shape[0]
cov_likelihood = np.diag(np.full(n, noise**2,dtype=np.float32))
y_distribution_fine = tda.GaussianLogLike(y_observed, cov_likelihood)


my_proposal = tda.GaussianRandomWalk(C=covariancep,scaling=1e-6, adaptive=True, gamma=1.1, period=10)
x_distribution = CustomUniform([0.01,0.5],[0.5,1.5])

if case == "1-step":
    # print("Err MF1: ",np.mean(np.abs(y_true-model_1(x_true))), flush=True)
    # print("Err MF2: ",np.mean(np.abs(y_true-model_2(x_true))), flush=True)
    # # print("Err MF3: ",np.mean(np.abs(y_true-model_3(x_true))), flush=True)
    # print("Err LF1: ",np.mean(np.abs(y_lf3-model_LF3(x_true))), flush=True)
    # print("Err LF2: ",np.mean(np.abs(y_lf2-model_LF2(x_true))), flush=True)
    # print("Err LF3: ",np.mean(np.abs(y_lf1-model_LF1(x_true))), flush=True)
    y_distribution_1 = tda.GaussianLogLike(y_observed+model_1(res.x)-model_3(res.x), cov_likelihood*scaling1)
    y_distribution_2 = tda.GaussianLogLike(y_observed+model_2(res.x)-model_3(res.x), cov_likelihood*scaling2)
    y_distribution_3 = tda.GaussianLogLike(y_observed, cov_likelihood*scaling3)

    my_posteriors = [ 
        tda.Posterior(x_distribution, y_distribution_1, model_1),
        tda.Posterior(x_distribution, y_distribution_2, model_2),
        tda.Posterior(x_distribution, y_distribution_2, model_3)
    ]

    level = 1
    
elif case == "MLDA":
    cov_lf1 = noise**2 * np.eye(y_lf1.shape[0])
    cov_lf2 = noise**2 * np.eye(y_lf2.shape[0])
    cov_lf3 = noise**2 * np.eye(y_lf3.shape[0])
    y_distribution_1 = tda.GaussianLogLike(y_lf1, cov_lf1*scaling1)
    y_distribution_2 = tda.GaussianLogLike(y_lf2, cov_lf2*scaling2)

    my_posteriors = [
        tda.Posterior(x_distribution, y_distribution_2, model_1), 
        tda.Posterior(x_distribution, y_distribution_1, model_2),
        tda.Posterior(x_distribution, y_distribution_1, model_3)
    ] 

    level = 1 

else:
    my_posteriors= tda.Posterior(x_distribution, y_distribution_fine, model_HF)
    level = 2

start_time = timeit.default_timer()
samples = tda.sample(my_posteriors, my_proposal, iterations=n_iter, n_chains=1,
                    initial_parameters=x_true, subchain_length=sub_sampling,store_coarse_chain=False)
elapsed_time = timeit.default_timer() - start_time

idata = tda.to_inference_data(samples, level='fine').sel(draw=slice(burnin, None, thin), groups="posterior")
ess = az.ess(idata)
print(ess)
mean_ess = np.mean([ess.data_vars[f'x{j}'].values for j in range(2)])
print(mean_ess)
az.plot_trace(idata)
plt.savefig("trace_2levels_"+case)
