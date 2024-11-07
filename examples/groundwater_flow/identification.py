# Standard Library Imports
import os
import sys
import timeit

# Optimize performance by setting environment variables
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Third-party Library Imports
import numpy as np
import pandas as pd
import tensorflow as tf
import arviz as az
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, multivariate_normal, uniform
from tensorflow.keras.models import load_model
import tinyDA as tda
from itertools import product
from numba import jit

# Local Module Imports
sys.path.append('./data/data_generation/')
from model import Model
from utils import *

case = "1-step"  # Options: "2-step"/"1-step"/"FOM"

# MCMC Parameters
noise        = 0.01
scaling      = 0.05
n_iter       = 125000
burnin       = 25000
thin         = 1000
sub_sampling = 2

# Initialize Parameters
n_samples = 25
np.random.seed(2109)
random_samples = np.random.randint(0, 160, n_samples)
n_eig = 64
X_values = np.loadtxt('data/X_test_h1.csv', delimiter=',')
y_values = np.loadtxt('data/y_test_h1.csv', delimiter=',')

# Resolution Parameters for Different Solvers
resolutions = [(50, 50), (25, 25), (10, 10)]
field_mean, field_stdev, lamb_cov, mkl = 1, 1, 0.1, 64

# Instantiate Models for Different Resolutions
solver_h1 = Model(resolutions[0], field_mean, field_stdev, mkl, lamb_cov)
solver_h2 = Model(resolutions[1], field_mean, field_stdev, mkl, lamb_cov)
solver_h3 = Model(resolutions[2], field_mean, field_stdev, 32, lamb_cov)

# Set Up Transmissivity Fields Using Mesh Coordinates
# Adjust the transmissivity based on h1 by retrieving mesh coordinates
coords_h1, coords_h2, coords_h3 = (
    np.array(solver.solver.mesh.coordinates()) for solver in (solver_h1, solver_h2, solver_h3)
)

# Define structured array data type for row-wise comparison
dtype = {
    'names': [f'f{i}' for i in range(coords_h1.shape[1])],
    'formats': [coords_h1.dtype] * coords_h1.shape[1]
}

# Convert mesh coordinates to structured arrays
structured_h1 = coords_h1.view(dtype)
structured_h2 = coords_h2.view(dtype)
structured_h3 = coords_h3.view(dtype)

# Create boolean vectors to identify matching rows between h1 and h2/h3 meshes
bool_vector2 = np.in1d(structured_h1, structured_h2)
bool_vector3 = np.in1d(structured_h1, structured_h3)

solver_h2.random_process.eigenvalues = solver_h1.random_process.eigenvalues
solver_h2.random_process.eigenvectors = solver_h1.random_process.eigenvectors[bool_vector2]
solver_h3.random_process.eigenvalues = solver_h1.random_process.eigenvalues
solver_h3.random_process.eigenvectors = solver_h1.random_process.eigenvectors[bool_vector3]

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

# Model Definitions for Different Cases
if case == "2-step":
    # Load Models for Low- and Multi-fidelity Predictions
    models_l = [load_model(f'models/single_fidelity/resolution_h1/samples_16000/model_fold_{i}.keras') for i in range(1, 5)]
    models_l_s = [load_model(f'models/single_fidelity/resolution_h3/samples_64000/model_fold_{i}.keras') for i in range(1, 5)]
    model_mf = load_model('models/multi_fidelity/resolution_50-10_2step/samples_16000/model_fold_4.keras')
    
    # Define TensorFlow Functions with JIT Compilation
    @tf.function(jit_compile=True)
    def model_lf(input):
        input_reshaped = tf.reshape(input, (1, 64))
        return tf.reduce_mean([mod(input_reshaped, training=False)[0] for mod in models_l], axis=0)

    @tf.function(jit_compile=True)
    def model_lf_s(input):
        input_reshaped = tf.reshape(input, (1, 64))
        return tf.reduce_mean([mod(input_reshaped, training=False)[0] for mod in models_l_s], axis=0)

    @tf.function(jit_compile=True)
    def model_hf(input):
        input_reshaped = tf.reshape(input, (1, 64))
        output_lf = tf.reshape(model_lf(input_reshaped), (1, 25))
        output_lf_s = tf.reshape(model_lf_s(input_reshaped), (1, 25))
        return model_mf([input_reshaped, output_lf_s, output_lf], training=False)[0]

    def model_LF(input): return model_lf(input).numpy().flatten()
    def model_HF(input): return model_hf(input).numpy().flatten()

elif case == "1-step":
    # Load Models for Low- and Multi-fidelity Predictions
    models_l = [load_model(f'models/single_fidelity/resolution_h1/samples_16000/model_fold_{i}.keras') for i in range(1, 5)]
    model_mf = load_model('models/multi_fidelity/resolution_50-25/samples_16000/model_fold_4.keras')

    # Define TensorFlow Functions with JIT Compilation
    @tf.function(jit_compile=True)
    def model_lf(input):
        input_reshaped = tf.reshape(input, (1, 64))
        return tf.reduce_mean([mod(input_reshaped, training=False)[0] for mod in models_l], axis=0)

    @tf.function(jit_compile=True) 
    def model_hf1step(input1, input2):
        input1    = tf.reshape(input1, (1, 64))
        input2    = tf.reshape(input2, (1, 25))
        output_lf = tf.reshape(model_lf(tf.reshape(input1, (1, 64))), (1, 25))
        return model_mf([input1,input2,output_lf], training=False)[0]
    
    def model_HF(input):
        coarse_data = tf.constant(solver_h2_data(input), dtype=tf.float32)
        return model_hf1step(input, coarse_data).numpy().flatten()

    def model_LF(input): return model_lf(input).numpy().flatten()

else:
    def model_HF(input): return solver_h1_data(input).flatten()

# Prior and Proposal Distributions
x_distribution = stats.multivariate_normal(mean=np.zeros(64), cov=np.eye(64))
my_proposal = tda.CrankNicolson(scaling=scaling, adaptive=False, gamma=1.01, period=100)
Times, Time_ESS, ESS = [], [], []

# Sampling for Each Random Sample
for i, sample in enumerate(random_samples, start=1):
    print(f'Sample = {sample}')
    x_true, y_true = X_values[sample], y_values[sample]
    y_observed = y_true + np.random.normal(scale=noise, size=y_true.shape[0])

    # Likelihood Distributions
    cov_likelihood = noise**2 * np.eye(25)
    y_distribution_coarse = tda.AdaptiveGaussianLogLike(y_observed, cov_likelihood)
    y_distribution_fine = tda.GaussianLogLike(y_observed, cov_likelihood)

    # Initialize Posteriors
    my_posteriors = [
        tda.Posterior(x_distribution, y_distribution_coarse, model_LF), 
        tda.Posterior(x_distribution, y_distribution_fine, model_HF)
    ] if case != "FOM" else tda.Posterior(x_distribution, y_distribution_fine, model_HF)

    # Run MCMC Sampling
    start_time = timeit.default_timer()
    samples = tda.sample(my_posteriors, my_proposal, iterations=n_iter, n_chains=1,
                         initial_parameters=np.zeros(64), subsampling_rate=sub_sampling,
                         adaptive_error_model='state-independent')
    elapsed_time = timeit.default_timer() - start_time

    # Effective Sample Size (ESS)
    idata = tda.to_inference_data(samples, level='fine').sel(draw=slice(burnin, None, thin), groups="posterior")
    ess = az.ess(idata)
    mean_ess = np.mean([ess.data_vars[f'x{j}'].values for j in range(64)])

    # Store Results
    Times.append(elapsed_time)
    ESS.append(mean_ess)
    Time_ESS.append(elapsed_time / mean_ess)

    print(f'Time: {elapsed_time:.2f}, ESS: {mean_ess:.2f}, Time/ESS: {elapsed_time / mean_ess:.2f} ({i}/{n_samples})')

# Save Results
output_folder = './data/recorded_values'
np.save(os.path.join(output_folder, f'MDA_MF_{case}_ratio_01.npy'), Time_ESS)
np.save(os.path.join(output_folder, f'MDA_MF_{case}_times_01.npy'), Times)
np.save(os.path.join(output_folder, f'MDA_MF_{case}_ESS_01.npy'), ESS)