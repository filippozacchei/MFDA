import os
import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from pyDOE import lhs as lhcs
from scipy.stats.distributions import norm as norm_dist
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import h5py

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
    field_mean, field_stdev, lamb_cov = 1, 1, 0.025
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


def project_to_pod_basis(coarse_data, n_components=45):
    """
    Perform Singular Value Decomposition (SVD) and project data onto the POD basis.
    
    :param coarse_data: Data matrix to be projected.
    :param n_components: Number of components to retain in the projection.
    :return: The retained POD basis.
    """
    y_t = coarse_data.T
    U, S, Vh = np.linalg.svd(y_t, full_matrices=False)

    return U, S, Vh

def solver_data(solver, x):
    """
    Run the solver for given input x and return the computed data.
    """
    solver.solve(x)
    return solver.get_solution()

def project_and_save_pod(solvers, solver_key, path_prefix):
    """
    Project solver data onto POD basis and save the results.
    
    :param solver_key: The key identifying the solver.
    :param samples: Samples.
    :param path_prefix: Directory path to save the results.
    """
    
    X_train = np.loadtxt(f"{path_prefix}/X_train_{solver_key}_100_01.csv", delimiter=",")
    n_samples = X_train.shape[0]
    y_train = np.array([solver_data(solvers[solver_key], X_train[i, :]) for i in tqdm(range(int(n_samples)), desc=f"Processing {solver_key} samples")])
    
    X_test = np.loadtxt(f"{path_prefix}/X_test_{solver_key}_100_01.csv", delimiter=",")
    n_samples = X_test.shape[0]
    y_test = np.array([solver_data(solvers[solver_key], X_test[i, :]) for i in tqdm(range(int(n_samples)), desc=f"Processing {solver_key} samples")])
    
    
    U, S, Vh = project_to_pod_basis(y_train)
       
    print(np.linalg.norm(y_train.T-U@(U.T@y_train.T)))
    print(np.linalg.norm(y_test.T-U@(U.T@y_test.T)))
    
    # Export the basis to an HDF5 file
    with h5py.File(f"pod_data_{solver_key}.h5", 'w') as h5file:
        # Save POD modes (spatial basis)
        h5file.create_dataset('POD_modes', data=U)  # Shape: (grid_points, num_modes)
        
        # Save singular values
        h5file.create_dataset('singular_values', data=S)  # Shape: (num_modes,)
        h5file.create_dataset('POD_coefficients', data=Vh)  # Shape: (num_modes,)
        h5file.create_dataset('train_solution', data=y_train)  # Shape: (num_modes,)
        h5file.create_dataset('test_solution', data=y_test)  # Shape: (num_modes,)
        
        # Save spatial grid
        h5file.attrs['num_modes'] = U.shape[1]  # Number of modes saved

print("POD basis exported to 'pod_basis.h5'.")

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

    print("\nProject solution and save POD \n")
    
    # Perform POD projection and save for all solvers
    for key in ["h1", "h2", "h3", "h4"]:
        project_and_save_pod(solvers, key, "../../data/data")

if __name__ == "__main__":
    main()
