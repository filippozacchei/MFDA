# Standard Library Imports
import os
import time
from timeit import default_timer as timer
import sys
import timeit
from scipy.optimize import least_squares
from timeit import repeat

# Third-party Library Imports
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as stats
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
os.environ['OMP_NUM_THREADS'] = '48'

from model import Model_DR
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.extend([os.path.join(BASE_DIR, 'forward_models'), os.path.join(BASE_DIR, 'utils')])
from data_utils import load_hdf5, prepare_lstm_dataset
from pod_utils import reshape_to_pod_2d_system_snapshots, project, reshape_to_lstm, reconstruct, reconstruct_eff, reconstruct_eff1

# Import necessary modules
from multi_fidelity_lstm import MultiFidelityLSTM

case = "1-step"
num_modes = 25
n_runs = 25

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
    num_samples, _, n, _ = u_data_coarse.shape

    if time_steps_coarse != time_steps_fine:
            # Normalize time grids
        time_coarse = np.linspace(0, 1, time_steps_coarse)
        time_fine = np.linspace(0, 1, time_steps_fine)

        # Reshape: merge spatial dimensions for vectorized processing
        reshaped_input = u_data_coarse.reshape(num_samples, time_steps_coarse, -1)  # shape: (samples, time, n*n)
        reshaped_output = np.zeros((num_samples, time_steps_fine, reshaped_input.shape[-1]))

        for sample_idx in range(num_samples):
            # Create interpolator for all spatial points at once
            interpolator = interp1d(time_coarse, reshaped_input[sample_idx], axis=0, kind='cubic', fill_value="extrapolate")
            reshaped_output[sample_idx] = interpolator(time_fine)
        

        # Reshape back to original spatial structure
        u_data_coarse_interpolated = reshaped_output.reshape(num_samples, time_steps_fine, n, n)
        return u_data_coarse_interpolated
    else:
        return u_data_coarse

def load_and_process_data(config, num_modes=25):
    """
    Load datasets, apply POD projection, and prepare LSTM-ready data.
    :param config: Configuration dictionary.
    :param num_modes: Number of POD modes to retain.
    :return: Processed training and testing data.
    """
    train_data = load_hdf5(config["train"])
    pod_basis = load_hdf5(config["pod_basis"])
    
    u_train_snapshots = reshape_to_pod_2d_system_snapshots(train_data['u'], train_data['v'])

    U, Sigma = pod_basis["POD_modes"], pod_basis["singular_values"]
    v_train = project(U, Sigma, u_train_snapshots, num_modes=num_modes)
    v_train_lstm = reshape_to_lstm(v_train, train_data['u'].shape[0], train_data['u'].shape[1], num_modes)

    X_train_init = np.column_stack((train_data["d1"], train_data["beta"]))
    X_train_prep = prepare_lstm_dataset(X_train_init, train_data["t"], v_train_lstm)
    X_train, y_train = X_train_prep[:, :, :3], X_train_prep[:, :, 3:]

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    y_train = scaler_Y.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)

    return scaler_X, scaler_Y

np.random.seed(123)
config_filepath = 'config/config_MultiFidelity_3_bis.json'
config = load_configuration(config_filepath)
scaler_X, scaler_Y = load_and_process_data(config, num_modes=num_modes)

# Instantiate Models for Different Resolutions
solver_h1 = Model_DR(n=128, dt=0.1, L=20., T=50.05)
solver_h2 = Model_DR(n=64,  dt=0.2, L=20., T=50.05)
solver_h3 = Model_DR(n=32,  dt=0.5, L=20., T=50.05)
solver_h4 = Model_DR(n=16,  dt=1.0, L=20., T=50.05)

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
    if time_steps_coarse != time_steps_fine:
        time_coarse = np.linspace(0, 1, time_steps_coarse)
        time_fine = np.linspace(0, 1, time_steps_fine)

        u_data_flat = u_data_coarse.reshape(n * n, -1)
        splines = interp1d(time_coarse, u_data_flat, kind='cubic', axis=1, fill_value="extrapolate")
        interpolated = splines(time_fine).reshape(n, n, time_steps_fine).transpose(2, 0, 1)
        return interpolated[np.newaxis]
    else:
        u_Data = u_data_coarse.transpose(2,0,1)
        return u_Data[np.newaxis]

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

def model_HF_lag(input):  return solver_data(solver_h1, input, lag=10).flatten()
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
    
if case == "1-step":
    # Load Models for Low- and Multi-fidelity Predictions
    models_1 = load_model(f'models/multi_fidelity_bis/resolution_h4/model.keras', safe_mode=False)
    models_2 = load_model(f'models/multi_fidelity_bis/resolution_h3-h4/model.keras', safe_mode=False)
    models_3 = load_model(f'models/multi_fidelity_bis/resolution_h2-h3-h4/model.keras', safe_mode=False)
    config_filepath = 'config/config_MultiFidelity_3_bis.json'
    config = load_configuration(config_filepath)
    pod_basis = load_hdf5(config["pod_basis"])
    pod_basis_coarse1 = load_hdf5(config["pod_basis_coarse1"])
    pod_basis_coarse2 = load_hdf5(config["pod_basis_coarse2"])
    pod_basis_coarse3 = load_hdf5(config["pod_basis_coarse3"])
    
    #@tf.function(jit_compile=True)
    def model_mf1(input1, input2):
        return models_1([input1,input2], training=False)[0]
    
    #@tf.function(jit_compile=True)
    def model_mf2(input1, input2, input3):
        return models_2([input1,input2,input3], training=False)[0]
    
    #@tf.function(jit_compile=True)
    def model_mf3(input1, input2, input3, input4):
        return models_3([input1,input2,input3,input4], training=False)[0]
    
    grid_points_h4 = 16*16
    grid_points_h3 = 32*32
    grid_points_h2 = 64*64
    grid_points_h1 = 128*128
    time_steps_coarseh4 = 51
    time_steps_coarseh3 = 101
    time_steps_coarseh2 = 251
    time_steps_fine = 501
    time_fine = np.arange(0, 50.05, 0.1)
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
        coarse_data = coarse_data.reshape(1, time_steps_fine, num_modes)*Sigma[:num_modes]
        print(coarse_data.shape)
        #coarse_data = (scaler_Y.transform(coarse_data.reshape(-1,coarse_data.shape[-1]))).reshape(coarse_data.shape) 
        return coarse_data

    def prepare_input(input_data, time_steps_fine):
        param_expanded = expand_parameters(input_data, time_steps_fine)
        X = np.concatenate([param_expanded, time_expanded], axis=2)
        return X

    def model_1(input_data):
        print(input_data.shape)
        coarse_data = prepare_coarse_data(solver_h4, U_coarse3, Sigma_coarse3, input_data, time_steps_coarseh4, grid_points_h4, num_modes)
        X = prepare_input(input_data, time_steps_fine)
        X = scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        predictions = model_mf1(X, coarse_data).numpy()
        print(predictions.shape)
        #predictions = scaler_Y.inverse_transform(predictions)
        sol_u, sol_v = reconstruct_eff1(U,Sigma,predictions/Sigma[:num_modes],25)
        sol = get_data(sol_u, sol_v, lag=10).flatten()
        return sol

    def model_2(input_data):
        coarse_data1 = prepare_coarse_data(solver_h4, U_coarse3, Sigma_coarse3, input_data, time_steps_coarseh4, grid_points_h4, num_modes)
        coarse_data2 = prepare_coarse_data(solver_h3, U_coarse2, Sigma_coarse2, input_data, time_steps_coarseh3, grid_points_h3, num_modes)
        X = prepare_input(input_data, time_steps_fine)
        X = scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        predictions = model_mf2(X, coarse_data2, coarse_data1).numpy()
        #predictions = scaler_Y.inverse_transform(predictions)
        sol_u, sol_v = reconstruct_eff1(U,Sigma,predictions/Sigma[:num_modes],25)
        sol = get_data(sol_u, sol_v, lag=10).flatten()
        return sol

    def model_3(input_data):
        coarse_data1 = prepare_coarse_data(solver_h4, U_coarse3, Sigma_coarse3, input_data, time_steps_coarseh4, grid_points_h4, num_modes)
        coarse_data2 = prepare_coarse_data(solver_h3, U_coarse2, Sigma_coarse2, input_data, time_steps_coarseh3, grid_points_h3, num_modes)
        coarse_data3 = prepare_coarse_data(solver_h2, U_coarse1, Sigma_coarse1, input_data, time_steps_coarseh2, grid_points_h2, num_modes)
        X = prepare_input(input_data, time_steps_fine)
        X = scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        predictions = model_mf3(X, coarse_data3, coarse_data2, coarse_data1).numpy()
        #predictions = scaler_Y.inverse_transform(predictions)
        sol_u, sol_v = reconstruct_eff1(U,Sigma,predictions/Sigma[:num_modes],25)
        sol = get_data(sol_u, sol_v, lag=10).flatten()
        return sol
        
    NUM_DATAPOINTS = 2
    input_data = np.array([0.05,0.5])
    

    # Measure execution time for model_1
    model_1_output = model_1(input_data)
    model_2_output = model_2(input_data)
    model_3_output = model_3(input_data)

x_distribution = CustomUniform([0.01,0.5],[0.1,1.5])

errors_max = {
    "MF1": [],
    "MF2": [],
    "MF3": [],
    "LF1": [],
    "LF2": [],
    "LF3": []
}

errors = {
    "MF1": [],
    "MF2": [],
    "MF3": [],
    "LF1": [],
    "LF2": [],
    "LF3": []
}

times = {
    "MF1": [],
    "MF2": [],
    "MF3": [],
    "LF1": [],
    "LF2": [],
    "LF3": []
}

for _ in range(n_runs):
    x_true_temp = x_distribution.rvs()
    y_hf_temp = model_HF_lag(x_true_temp)

    start = time.perf_counter()
    y_mf1_temp = model_1(x_true_temp)
    times["MF1"].append(time.perf_counter() - start)
    
    start = time.perf_counter()
    y_mf2_temp = model_2(x_true_temp)
    times["MF2"].append(time.perf_counter() - start)
    
    start = time.perf_counter()
    y_mf3_temp = model_3(x_true_temp)
    times["MF3"].append(time.perf_counter() - start)

    start = time.perf_counter()
    y_lf1_temp = model_LF3_lag(x_true_temp)
    times["LF1"].append(time.perf_counter() - start)
    
    start = time.perf_counter()
    y_lf2_temp = model_LF2_lag(x_true_temp)
    times["LF2"].append(time.perf_counter() - start)
    
    start = time.perf_counter()
    y_lf3_temp = model_LF1_lag(x_true_temp)
    times["LF3"].append(time.perf_counter() - start)
    
    errors["MF1"].append(np.sqrt(np.mean((y_hf_temp - y_mf1_temp)**2)))
    errors["MF2"].append(np.sqrt(np.mean((y_hf_temp - y_mf2_temp)**2)))
    errors["MF3"].append(np.sqrt(np.mean((y_hf_temp - y_mf3_temp)**2)))
    errors["LF1"].append(np.sqrt(np.mean((y_hf_temp - y_lf1_temp)**2)))
    errors["LF2"].append(np.sqrt(np.mean((y_hf_temp - y_lf2_temp)**2)))
    errors["LF3"].append(np.sqrt(np.mean((y_hf_temp - y_lf3_temp)**2))) 
    errors_max["MF1"].append(np.max(np.sqrt((y_hf_temp - y_mf1_temp)**2)))
    errors_max["MF2"].append(np.max(np.sqrt((y_hf_temp - y_mf2_temp)**2)))
    errors_max["MF3"].append(np.max(np.sqrt((y_hf_temp - y_mf3_temp)**2)))
    errors_max["LF1"].append(np.max(np.sqrt((y_hf_temp - y_lf1_temp)**2)))
    errors_max["LF2"].append(np.max(np.sqrt((y_hf_temp - y_lf2_temp)**2)))
    errors_max["LF3"].append(np.max(np.sqrt((y_hf_temp - y_lf3_temp)**2)))

for key in errors:
    mean_err = np.mean(errors[key])
    std_err = np.std(errors[key])
    print(f"{key}: Mean MAE = {mean_err:.4e}, Std = {std_err:.4e}")

for key in errors:
    mean_max_err = np.mean(errors_max[key])
    std_max_err = np.std(errors_max[key])
    print(f"{key}: Max MAE = {mean_max_err:.4e}, Std = {std_max_err:.4e}")

for key in times:
    mean_time = np.mean(times[key])
    std_time = np.std(times[key])
    print(f"{key}: Mean time = {mean_time:.4e} s, Std = {std_time:.4e} s")


