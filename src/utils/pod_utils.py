import h5py
import numpy as np
import logging
from sklearn.utils.extmath import randomized_svd as r_svd
from scipy.linalg import svd

def reshape_to_pod_2d_snapshots(data):
    """Reshapes the dataset into a 2D snapshot matrix."""
    num_samples, time_steps, n, _ = data.shape
    grid_points = n * n
    return data[:num_samples, :].reshape(num_samples * time_steps, grid_points).T

def reshape_to_pod_2d_system_snapshots(data1, data2):
    """Optimized reshaping of dataset into a 2D snapshot matrix."""
    num_samples, time_steps, n, _ = data1.shape
    num_snapshots = num_samples * time_steps
    grid_points = n * n
    reshaped_data1 = data1.reshape(num_snapshots, grid_points).T  # Shape (grid_points, num_snapshots)
    reshaped_data2 = data2.reshape(num_snapshots, grid_points).T  # Shape (grid_points, num_snapshots)
    return np.concatenate((reshaped_data1, reshaped_data2), axis=0)  # Shape (2 * grid_points, num_snapshots)

def reshape_to_lstm(data, num_samples: int, time_steps: int, num_modes: int):
    """Reshapes the dataset into a 2D snapshot matrix."""
    return data.reshape(num_samples, time_steps, num_modes)

def compute_svd(snapshots, n_components=100, method='standard'):
    """
    Computes the Proper Orthogonal Decomposition (POD) using SVD.
    
    :param snapshots: Input data matrix (features x samples).
    :param n_components: Number of singular values and vectors to compute.
    :param method: SVD method ('standard' for full SVD, 'truncated' for economy SVD).
    :return: Tuple (U, Sigma, VT) where U contains left singular vectors,
             Sigma contains singular values, and VT contains right singular vectors.
    """
    logging.info(f"Computing SVD using method: {method}")
    
    if method == 'standard':
        U, Sigma, VT = svd(snapshots, full_matrices=False)  # More memory efficient
    elif method == 'truncated':
        U, Sigma, VT = svd(snapshots, full_matrices=False)
        U, Sigma, VT = U[:, :n_components], Sigma[:n_components], VT[:n_components, :]
    else:
        raise ValueError("Invalid method. Choose 'standard' or 'truncated'.")
    
    return U, Sigma, VT.T

def randomized_svd(snapshots, n_components=100):
    """Computes the Proper Orthogonal Decomposition (POD) using SVD."""
    U, Sigma, VT = r_svd(snapshots, n_components=n_components, random_state=42)
    return U, Sigma, VT.T  # Return transposed time coefficients

def reconstruct(U, sigma, time_coefficients, num_modes, original_snapshots=None):
    """Reconstructs data using POD basis and computes error efficiently."""
    reconstructed = U[:,:num_modes] @ (time_coefficients * sigma[:num_modes]).T  # Element-wise multiplication
    if original_snapshots is not None:
        error_rmse = np.mean(np.linalg.norm(original_snapshots - reconstructed,axis=0,ord=1)/np.linalg.norm(original_snapshots,axis=0,ord=1))
        return reconstructed, error_rmse 
    return reconstructed

def project(U, Sigma, snapshots,num_modes):
    return (U[:,:num_modes].T@snapshots/Sigma[:num_modes,None]).T

def load_svd_data(filename: str):
    """
    Loads SVD data (POD modes and singular values) from an HDF5 file.
    :param filename: Path to the HDF5 file containing the SVD data.
    :return: Tuple (POD_modes, singular_values)
    """
    try:
        logging.info(f"Loading SVD data from {filename}")
        with h5py.File(filename, 'r') as h5file:
            POD_modes = h5file['POD_modes'][:]
            singular_values = h5file['singular_values'][:]
        return POD_modes, singular_values
    except FileNotFoundError:
        logging.error(f"SVD data file not found: {filename}")
        raise
    except Exception as e:
        logging.error(f"Error loading SVD data from {filename}: {e}")
        raise
