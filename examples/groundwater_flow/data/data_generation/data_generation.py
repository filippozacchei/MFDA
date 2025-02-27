import os
import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from pyDOE import lhs as lhcs
from scipy.stats.distributions import norm as norm_dist
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

# Environment Setup
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Import your custom model module (assuming correct structure)
from model import Model

def setup_environment():
    """
    Configure the random seed and return the setup parameters.
    """
    np.random.seed(123)
    n_samples = 128000
    resolutions = [(100, 100), (50, 50), (25,25), (10, 10)]
    field_mean, field_stdev, lamb_cov = 1, 1, 0.1
    mkl_values = [64, 64, 64, 64]
    x_data = y_data = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    datapoints = np.array(list(product(x_data, y_data)))
    
    return n_samples, resolutions, field_mean, field_stdev, lamb_cov, mkl_values, datapoints

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

def solver_data(solver, datapoints, x):
    """
    Run the solver for given input x and return the computed data.
    """
    solver.solve(x)
    return solver.get_data(datapoints)

def generate_samples(n_samples):
    return norm_dist(loc=0, scale=1).ppf(lhcs(64, samples=n_samples))
    

def generate_solver_data(solvers, solver_key, samples, datapoints, path_prefix):
    """
    Generate data for a specific solver and save training/testing sets.
    
    :param solver_key: The key identifying the solver in the `solvers` dictionary.
    :param samples: previously generated samples.
    :param path_prefix: Directory path to save the generated data.
    """
    n_samples = samples.shape[0]
    solver = solvers[solver_key]
    data = np.zeros((n_samples, len(datapoints)))
    
    for i in tqdm(range(n_samples), desc=f"Processing {solver_key} samples"):
        data[i, :] = solver_data(solver, datapoints, samples[i, :])
    
    # Split data into training and testing sets
    split_idx = int(0.9 * n_samples)
    np.savetxt(f"{path_prefix}/X_train_{solver_key}_100_01.csv", samples[:split_idx], delimiter=",")
    np.savetxt(f"{path_prefix}/y_train_{solver_key}_100_01.csv", data[:split_idx], delimiter=",")
    np.savetxt(f"{path_prefix}/X_test_{solver_key}_100_01.csv", samples[split_idx:], delimiter=",")
    np.savetxt(f"{path_prefix}/y_test_{solver_key}_100_01.csv", data[split_idx:], delimiter=",")

def project_to_pod_basis(coarse_data, n_components=45):
    """
    Perform Singular Value Decomposition (SVD) and project data onto the POD basis.
    
    :param coarse_data: Data matrix to be projected.
    :param n_components: Number of components to retain in the projection.
    :return: The retained POD basis.
    """
    y_t = coarse_data.T
    U, S, Vh = np.linalg.svd(y_t, full_matrices=False)

    return U[:, :n_components]

def project_and_save_pod(solvers, solver_key, samples, datapoints, path_prefix):
    """
    Project solver data onto POD basis and save the results.
    
    :param solver_key: The key identifying the solver.
    :param samples: Samples.
    :param path_prefix: Directory path to save the results.
    """
    n_samples = samples.shape[0]
    X_train = np.loadtxt(f"{path_prefix}/X_train_{solver_key}_100_01.csv", delimiter=",")
    coarse_sol_train = np.array([solver_data(solvers[solver_key], datapoints, X_train[i, :]) for i in tqdm(range(int(n_samples * 0.9)), desc=f"Processing {solver_key} samples")])
    
    basis = project_to_pod_basis(coarse_sol_train)
    X_train_pod = coarse_sol_train @ basis
    
    X_test = np.loadtxt(f"{path_prefix}/X_test_{solver_key}_100_01.csv", delimiter=",")
    coarse_sol_test = np.array([solver_data(solvers[solver_key], datapoints, X_test[i, :]) for i in tqdm(range(int(n_samples * 0.1)), desc=f"Processing {solver_key} samples")])
    X_test_pod = coarse_sol_test @ basis
    
    # Save projected data and basis
    np.savetxt(f"{path_prefix}/X_train_{solver_key}_pod_100_01.csv", X_train_pod, delimiter=",")
    np.savetxt(f"{path_prefix}/X_test_{solver_key}_pod_100_01.csv", X_test_pod, delimiter=",")
    np.savetxt(f"{path_prefix}/POD_basis_{solver_key}_100_01.csv", basis, delimiter=",")

def print_simulation_parameters(n_samples, resolutions, field_mean, field_stdev, lamb_cov, mkl_values):
    """
    Print the simulation parameters to the screen for logging purposes.
    
    :param n_samples: Number of samples used in the simulation.
    :param resolutions: List of resolutions used in the simulation.
    :param field_mean: Mean value of the field.
    :param field_stdev: Standard deviation of the field.
    :param lamb_cov: Covariance of the lambda parameter.
    :param mkl_values: List of MKL values for the solvers.
    """
    print("Simulation Parameters")
    print("=====================")
    print(f"Number of samples: {n_samples}")
    print(f"Resolutions h_i: {resolutions}")
    print(f"Field Mean: {field_mean}")
    print(f"Field Standard Deviation: {field_stdev}")
    print(f"Lambda Covariance: {lamb_cov}")
    print(f"MKL Values: {mkl_values}")
    print("=====================")

def main():

    print("\nSetup equation solvers \n")

    # Setup environment and solvers
    n_samples, resolutions, field_mean, field_stdev, lamb_cov, mkl_values, datapoints = setup_environment()

    print_simulation_parameters(n_samples, resolutions, field_mean, field_stdev, lamb_cov, mkl_values)

    solvers = {
        "h1": Model(resolutions[0], field_mean, field_stdev, mkl_values[0], lamb_cov),
        "h2": Model(resolutions[1], field_mean, field_stdev, mkl_values[1], lamb_cov),
        "h3": Model(resolutions[2], field_mean, field_stdev, mkl_values[2], lamb_cov),
        "h4": Model(resolutions[3], field_mean, field_stdev, mkl_values[2], lamb_cov)
    }    

    # Setup random processes between solvers
    setup_random_process(solvers["h1"], solvers["h2"])
    setup_random_process(solvers["h1"], solvers["h3"])
    setup_random_process(solvers["h1"], solvers["h4"])

    print("\nGenerate solver data \n")

    # Generate data for all solvers
    samples = generate_samples(n_samples)

    for key in ["h1", "h2", "h3", "h4"]:
        generate_solver_data(solvers, key, samples, datapoints, "../../data")


if __name__ == "__main__":
    main()
