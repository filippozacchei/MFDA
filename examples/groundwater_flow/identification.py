# Standard Library Imports
import os
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

# Local Module Imports
sys.path.append('./data/data_generation/')
import tinyDA as tda

from model import Model
# from utils import *

case = "MFDA2"  # "MFDA2", "MLDA", "MLDA2", "MFDA1, "FOM"

# MCMC Parameters
noise        = 0.001
noise_Str    = str(noise).replace('.', '_')
scaling      = 0.001
scale        = str(scaling).replace('.', '_')
scaling1     = 1
scaling2     = 1
scaling3     = 1
n_iter       = 26000
burnin       = 1000
thin         = 1
sub          = 1

sub_sampling = [5,sub]
n_train_samples = 64000

# Initialize Parameters
n_samples = 0
np.random.seed(31321)
random_samples = np.random.randint(0, 10000, n_samples)
n_eig = 64
X_values = np.loadtxt('data/data/X_test_h1_100_01.csv', delimiter=',')
y_values = np.loadtxt('data/data/y_test_h1_100_01.csv', delimiter=',')

# Resolution Parameters for Different Solvers
resolutions = [(100, 100), (50, 50), (25, 25), (10, 10)]
field_mean, field_stdev, lamb_cov, mkl = 1, 1, 0.1, 64 

# Instantiate Models for Different Resolutions
solver_h1 = Model(resolutions[0], field_mean, field_stdev, mkl, lamb_cov)
solver_h2 = Model(resolutions[1], field_mean, field_stdev, mkl, lamb_cov)
solver_h3 = Model(resolutions[2], field_mean, field_stdev, mkl, lamb_cov)
solver_h4 = Model(resolutions[3], field_mean, field_stdev, mkl, lamb_cov)

def setup_random_process(solver_high, solver_low):
    """
    Synchronize the random processes between the higher and lower fidelity models
    by matching transmissivity fields across different resolutions.
    """
    coords_high = solver_high.solver.mesh.coordinates()
    coords_low = solver_low.solver.mesh.coordinates()
    
    structured_high = np.array(coords_high).view([('', coords_high.dtype)] * coords_high.shape[1])
    structured_low = np.array(coords_low).view([('', coords_low.dtype)] * coords_low.shape[1])
    
    bool_mask = np.in1d(structured_high, structured_low)
    solver_low.random_process.eigenvalues = solver_high.random_process.eigenvalues
    solver_low.random_process.eigenvectors = solver_high.random_process.eigenvectors[bool_mask]  

# Setup random processes between solvers
setup_random_process(solver_h1, solver_h2)
setup_random_process(solver_h1, solver_h3)
setup_random_process(solver_h1, solver_h4)

# Define Points for Data Extraction
x_data = y_data = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
datapoints = np.array(list(product(x_data, y_data)))

# Solver Data Functions
def solver_h1_data(x):
    solver_h1.solve(x)
    return solver_h1.get_data(datapoints)
def solver_h2_data(x):
    solver_h2.solve(x) 
    return solver_h2.get_data(datapoints)
def solver_h3_data(x): 
    solver_h3.solve(x)
    return solver_h3.get_data(datapoints)
def solver_h4_data(x): 
    solver_h4.solve(x)
    return solver_h4.get_data(datapoints)

def model_HF(input): return solver_h1_data(input).flatten()
def model_LF1(input): return solver_h2_data(input).flatten()
def model_LF2(input): return solver_h3_data(input).flatten()
def model_LF3(input): return solver_h4_data(input).flatten()

# Model Definitions for Different Cases
if case == "MFDA2":
    # Load Models for Low- and Multi-fidelity Predictions
        # Load Models for Low- and Multi-fidelity Predictions
    model_nn = load_model(f'models/single_fidelity_100/resolution_10/samples_{n_train_samples}/model_fold_1.keras')
    models_1 = load_model(f'models/multi_fidelity_100_2step/input_10/samples_{n_train_samples}/model_fold_1.keras')
    models_2 = load_model(f'models/multi_fidelity_100_2step/input_10_25/samples_{n_train_samples}/model_fold_1.keras')
    models_3 = load_model(f'models/multi_fidelity_100_2step/input_10_25_50/samples_{n_train_samples}/model_fold_1.keras')
    
    @tf.function(jit_compile=True) 
    def model_mf1(input1, input2):
        input1    = tf.reshape(input1, (1, 64))
        input2    = tf.reshape(input2, (1, 25))
        return models_1([input1,input2], training=False)[0]
    
    @tf.function(jit_compile=True) 
    def model_mf2(input1, input2, input3):
        input1    = tf.reshape(input1, (1, 64))
        input2    = tf.reshape(input2, (1, 25))
        input3    = tf.reshape(input3, (1, 25))
        return models_2([input1,input2,input3], training=False)[0]
    
    @tf.function(jit_compile=True) 
    def model_mf3(input1, input2, input3, input4):
        input1    = tf.reshape(input1, (1, 64))
        input2    = tf.reshape(input2, (1, 25))
        input3    = tf.reshape(input3, (1, 25))
        input4    = tf.reshape(input4, (1, 25))
        return models_3([input1,input2,input3,input4], training=False)[0]
    
    # def model_1(input):
    #     coarse_data1 = tf.constant(solver_h4_data(input), dtype=tf.float32)
    #     return model_mf1(input, coarse_data1).numpy().flatten()
    @tf.function(jit_compile=True) 
    def model_cc(input):
        inputx = tf.reshape(input, (1, 64))
        return model_nn(inputx, training=False)[0]
    
    def model_1(input):
        coarse_data1 = model_cc(input)
        return model_mf1(input, coarse_data1).numpy().flatten()
    
    def model_2(input):
        coarse_data1 = model_cc(input)
        coarse_data2 = tf.constant(solver_h3_data(input), dtype=tf.float32)
        return model_mf2(input, coarse_data2, coarse_data1).numpy().flatten()

    def model_3(input):
        coarse_data2 = tf.constant(solver_h3_data(input), dtype=tf.float32)
        coarse_data3 = tf.constant(solver_h2_data(input), dtype=tf.float32)
        coarse_data1 = model_cc(input)
        return model_mf3(input, coarse_data3, coarse_data2,coarse_data1).numpy().flatten()
    
    def model_HF(input): return solver_h1_data(input).flatten()
    
    # Generate input data
    NUM_DATAPOINTS = 64
    input_data = np.random.normal(size=(NUM_DATAPOINTS, 1))
    input1 = input_data

    input2 = solver_h4_data(input1)
    input3 = solver_h3_data(input1)
    input4 = solver_h2_data(input1)
    
    execution_times = {
        "LF1": np.mean(repeat(lambda: solver_h4_data(input1), number=1, repeat=100)),
        "LF2": np.mean(repeat(lambda: solver_h3_data(input1), number=1, repeat=100)),
        "LF3": np.mean(repeat(lambda: solver_h2_data(input1), number=1, repeat=100)),
        "HF": np.mean(repeat(lambda: solver_h1_data(input1), number=1, repeat=100)),
    }

    for key, time in execution_times.items():
        print(f"Execution time {key}: {time}")

    execution_times = {
        "MF3": np.mean(repeat(lambda: model_mf3(input1, input2, input3, input4), number=1, repeat=100)),
        "MF2": np.mean(repeat(lambda: model_mf2(input1, input2, input3), number=1, repeat=100)),
        "MF1": np.mean(repeat(lambda: model_mf1(input1, input2), number=1, repeat=100)),
    }

    for key, time in execution_times.items():
        print(f"Execution time {key}: {time}")

    true_output = solver_h1_data(input1)
    errors = {
        "MF3": np.mean(np.sqrt((true_output - model_3(input1))**2)),
        "MF2": np.mean(np.sqrt((true_output - model_2(input1))**2)),
        "MF1": np.mean(np.sqrt((true_output - model_1(input1))**2)),
        "LF3": np.mean(np.sqrt((true_output - solver_h2_data(input1))**2)),
        "LF2": np.mean(np.sqrt((true_output - solver_h3_data(input1))**2)),
        "LF1": np.mean(np.sqrt((true_output - solver_h4_data(input1))**2)),
    }

    for key, err in errors.items():
        print(f"Error {key}: {err}")

elif case == "MFDA1":
    # Load Models for Low- and Multi-fidelity Predictions
    models_1 = load_model(f'models/multi_fidelity_100/input_10/samples_{n_train_samples}/model_fold_1.keras')
    models_2 = load_model(f'models/multi_fidelity_100/input_10_25/samples_{n_train_samples}/model_fold_1.keras')
    models_3 = load_model(f'models/multi_fidelity_100/input_10_25_50/samples_{n_train_samples}/model_fold_1.keras')

    @tf.function(jit_compile=True) 
    def model_mf1(input1, input2):
        input1    = tf.reshape(input1, (1, 64))
        input2    = tf.reshape(input2, (1, 25))
        return models_1([input1,input2], training=False)[0]
    
    @tf.function(jit_compile=True) 
    def model_mf2(input1, input2, input3):
        input1    = tf.reshape(input1, (1, 64))
        input2    = tf.reshape(input2, (1, 25))
        input3    = tf.reshape(input3, (1, 25))
        return models_2([input1,input2,input3], training=False)[0]
    
    @tf.function(jit_compile=True) 
    def model_mf3(input1, input2, input3, input4):
        input1    = tf.reshape(input1, (1, 64))
        input2    = tf.reshape(input2, (1, 25))
        input3    = tf.reshape(input3, (1, 25))
        input4    = tf.reshape(input4, (1, 25))
        return models_3([input1,input2,input3,input4], training=False)[0]
    
    # def model_1(input):
    #     coarse_data1 = tf.constant(solver_h4_data(input), dtype=tf.float32)
    #     return model_mf1(input, coarse_data1).numpy().flatten()
    
    def model_1(input):
        coarse_data1 = tf.constant(solver_h4_data(input), dtype=tf.float32)
        return model_mf1(input, coarse_data1).numpy().flatten()
    
    def model_2(input):
        coarse_data1 = tf.constant(solver_h4_data(input), dtype=tf.float32)
        coarse_data2 = tf.constant(solver_h3_data(input), dtype=tf.float32)
        return model_mf2(input, coarse_data2, coarse_data1).numpy().flatten()

    def model_3(input):
        coarse_data1 = tf.constant(solver_h4_data(input), dtype=tf.float32)
        coarse_data2 = tf.constant(solver_h3_data(input), dtype=tf.float32)
        coarse_data3 = tf.constant(solver_h2_data(input), dtype=tf.float32)
        return model_mf3(input, coarse_data3, coarse_data2,coarse_data1).numpy().flatten()
    
    def model_HF(input): return solver_h1_data(input).flatten()
    
    # Generate input data
    NUM_DATAPOINTS = 64
    input_data = np.random.normal(size=(NUM_DATAPOINTS, 1))
    input1 = input_data

    input2 = solver_h4_data(input1)
    input3 = solver_h3_data(input1)
    input4 = solver_h2_data(input1)

    execution_times = {
        "MF3": np.mean(repeat(lambda: model_mf3(input1, input2, input3, input4), number=1, repeat=1)),
        "MF2": np.mean(repeat(lambda: model_mf2(input1, input2, input3), number=1, repeat=1)),
        "MF1": np.mean(repeat(lambda: model_mf1(input1, input2), number=1, repeat=1)),
    }

    for key, time in execution_times.items():
        print(f"Execution time {key}: {time}")

    true_output = solver_h1_data(input1)
    errors = {
        "MF3": np.mean(np.sqrt((true_output - model_3(input1))**2)),
        "MF2": np.mean(np.sqrt((true_output - model_2(input1))**2)),
        "MF1": np.mean(np.sqrt((true_output - model_1(input1))**2)),
        "LF3": np.mean(np.sqrt((true_output - solver_h2_data(input1))**2)),
        "LF2": np.mean(np.sqrt((true_output - solver_h3_data(input1))**2)),
        "LF1": np.mean(np.sqrt((true_output - solver_h4_data(input1))**2)),
    }

    for key, err in errors.items():
        print(f"Error {key}: {err}")
    
elif case == "MLDA1":
    
    model_1 = model_LF2
    model_2 = model_LF1
    model_3 = model_HF
    
elif case == "MLDA2":
    
    model_1 = model_LF3
    model_2 = model_LF2
    model_3 = model_LF1


# Prior and Proposal Distributions
x_distribution = stats.multivariate_normal(mean=np.zeros(64), cov=np.eye(64))
Times, Time_ESS, ESS, samples_tot, ERR = [], [], [], [], []

# Sampling for Each Random Sample
for i, sample in enumerate(random_samples, start=1):
    print(f'Sample = {sample}')
    x_true = X_values[sample]
    y_true = y_values[sample]
    y_observed = y_true + np.random.normal(scale=noise, size=y_true.shape[0])
    if case != "FOM":
        print(f"\nMSE coarse simulation 1 test:  {np.sqrt(np.mean((model_1(x_true) - y_true)**2)):.4e}")
        print(f"\nMSE coarse simulation 2 test:  {np.sqrt(np.mean((model_2(x_true) - y_true)**2)):.4e}")
        print(f"\nMSE coarse simulation 3 test:  {np.sqrt(np.mean((model_3(x_true) - y_true)**2)):.4e}")
        print(f"\nMSE coarse simulation HF test:  {np.sqrt(np.mean((model_HF(x_true) - y_values[sample])**2)):.4e}")
        
    def ls(x):
        return (y_true-model_HF(x))

    res = least_squares(ls,np.zeros_like((x_true)), jac='3-point')
    covariance = np.linalg.pinv(res.jac.T @ res.jac)
    covariance *= 1/np.max(np.abs(covariance))
    print(covariance[1:5,1:5])

    # Likelihood Distributions
    cov_likelihood = noise**2 * np.eye(25)
    y_distribution_1 = tda.AdaptiveGaussianLogLike(y_observed, cov_likelihood*scaling1)
    y_distribution_2 = tda.AdaptiveGaussianLogLike(y_observed, cov_likelihood*scaling2)
    y_distribution_3 = tda.AdaptiveGaussianLogLike(y_observed, cov_likelihood*scaling3)
    y_distribution_fine = tda.GaussianLogLike(y_observed, cov_likelihood)
    my_proposal = tda.GaussianRandomWalk(C=covariance,scaling=scaling, adaptive=True, gamma=1.1, period=100)
    
    # Initialize Posteriors
    my_posteriors = [
        tda.Posterior(x_distribution, y_distribution_1, model_1), 
        tda.Posterior(x_distribution, y_distribution_2, model_2),
        tda.Posterior(x_distribution, y_distribution_3, model_3)
    ] if case != "FOM" else tda.Posterior(x_distribution, y_distribution_fine, model_HF)

    # Run MCMC Sampling
    start_time = timeit.default_timer()
    samples = tda.sample(my_posteriors, my_proposal, iterations=n_iter, n_chains=1,
                            initial_parameters=res.x, subchain_length=sub_sampling,
                            adaptive_error_model='state-independent',store_coarse_chain=False)
    elapsed_time = timeit.default_timer() - start_time

    # Effective Sample Size (ESS)
    idata = tda.to_inference_data(samples, level=2).sel(draw=slice(burnin, None, thin), groups="posterior")
    ess = az.ess(idata)
    mean_ess = np.mean([ess.data_vars[f'x{j}'].values for j in range(64)])

    # Store Results
    Times.append(elapsed_time)
    ESS.append(mean_ess)
    Time_ESS.append(elapsed_time / mean_ess)
    post = idata.posterior
    val=post.mean().to_array()
    err=(np.mean(np.sqrt((x_true-val)**2)))
    ERR.append(err)
    print(f'Time: {elapsed_time:.2f}, ESS: {mean_ess:.2f}, Time/ESS: {elapsed_time / mean_ess:.2f}, Err: {err:.3f} ({i}/{n_samples})')
    

    
# Save Results
output_folder = './data/recorded_values'
np.save(os.path.join(output_folder, f'MDA_MF_{case}_noise_{noise_Str}_ratio_sub{sub}_scaling{scale}.npy'), Time_ESS)
np.save(os.path.join(output_folder, f'MDA_MF_{case}_noise_{noise_Str}_times_sub{sub}_scaling{scale}.npy'), Times)
np.save(os.path.join(output_folder, f'MDA_MF_{case}_noise {noise_Str}_err_sub{sub}_scaling{scale}.npy'), ERR)
np.save(os.path.join(output_folder, f'MDA_MF_{case}_noise_{noise_Str}_ESS_sub{sub}_scaling{scale}.npy'), ESS)
