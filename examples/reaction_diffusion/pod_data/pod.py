import sys
import os
import json
import logging
import tensorflow as tf
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the 'models' directory to the Python path
sys.path.append(os.path.abspath('../../../src/utils/'))
sys.path.append(os.path.abspath('./examples/reaction_diffusion/data/'))

from data_utils import load_hdf5, save_hdf5
from pod_utils import *
from plot_utils import *

###############################################################################
# 1. Down-sampling utilities
###############################################################################
def block_average_2d(array2d, ratio):
    """
    Restricts a 2D (n_fine x n_fine) array to a coarser grid by block averaging.
    array2d: 2D numpy array of shape (n_fine, n_fine)
    ratio: integer down-sampling ratio, e.g. n_fine // n_coarse
    Returns:
      2D array of shape (n_fine//ratio, n_fine//ratio)
    """
    n_fine = array2d.shape[0]
    n_coarse = n_fine // ratio
    coarse_array = np.zeros((n_coarse, n_coarse), dtype=array2d.dtype)

    for I in range(n_coarse):
        for J in range(n_coarse):
            block = array2d[I*ratio:(I+1)*ratio, J*ratio:(J+1)*ratio]
            coarse_array[I, J] = np.mean(block)

    return coarse_array

def downsample_2d_system_pod_modes(U_fine, n_fine, n_coarse):
    """
    Down-sample (restrict) each 2D system POD mode from (n_fine x n_fine) 
    for each of 2 variables (u, v), into (n_coarse x n_coarse).
    U_fine: 2D numpy array of shape ((n_fine^2)*2, r), 
            where the first n_fine^2 rows correspond to the 'u' field 
            and the next n_fine^2 rows correspond to the 'v' field.
    n_fine: fine-grid dimension
    n_coarse: coarse-grid dimension (must divide n_fine for simple block averaging)
    Returns:
      U_coarse: 2D numpy array of shape ((n_coarse^2)*2, r)
    """
    ratio = n_fine // n_coarse
    r = U_fine.shape[1]  # number of POD modes retained

    # Prepare container for coarse modes
    U_coarse = np.zeros((2 * n_coarse * n_coarse, r), dtype=U_fine.dtype)

    for mode_idx in range(r):
        # Extract the 'u' portion of this mode and reshape to 2D
        mode_u = U_fine[0:n_fine*n_fine, mode_idx].reshape(n_fine, n_fine)
        # Extract the 'v' portion and reshape to 2D
        mode_v = U_fine[n_fine*n_fine:, mode_idx].reshape(n_fine, n_fine)

        # Block average each one
        mode_u_coarse = block_average_2d(mode_u, ratio)
        mode_v_coarse = block_average_2d(mode_v, ratio)

        # Flatten each coarse mode and store back
        U_coarse[0:n_coarse*n_coarse, mode_idx] = mode_u_coarse.flatten()
        U_coarse[n_coarse*n_coarse:, mode_idx] = mode_v_coarse.flatten()

    return U_coarse


###############################################################################
# 2. Main POD pipeline
###############################################################################
resolution = 'h1'

def get_output_name(resolution=''):
    """Generates the output filename for POD basis."""
    return f'pod_basis_system_{resolution}.h5'

def get_data_filenames(data_path, resolution=''):
    """Generates training and testing dataset filenames based on resolution."""
    return (
        os.path.join(data_path, f'reaction_diffusion_training_{resolution}.h5')
    )

logging.info("Starting POD processing pipeline.")

data_path = "../data/"

# Generate output filename
output_name = get_output_name(resolution)

# Load datasets
logging.info("Loading training and testing datasets.")
train_filename = get_data_filenames(data_path, resolution)
train_data = load_hdf5(train_filename)
# Reshape data into 2D snapshots
logging.info("Reshaping training and testing data.")
# You can limit how many samples you load, e.g. [:20], or adjust as needed
u_train_snapshots = reshape_to_pod_2d_system_snapshots(train_data['u'][:20], train_data['v'][:20])
u_test_snapshots = reshape_to_pod_2d_system_snapshots(train_data['u'][20:], train_data['v'][20:])

# 3. Compute POD using randomized SVD
logging.info("Computing POD basis using randomized SVD.")
U_fine, Sigma, VT = randomized_svd(u_train_snapshots, n_components=100)

# Project test data onto the same basis
VT_test = project(U_fine, Sigma, u_test_snapshots, num_modes=100)

# 5. Save computed POD basis (fine grid)
logging.info("Saving POD basis and singular values to HDF5 file.")
save_hdf5(output_name, 
            {'POD_modes': U_fine, 'singular_values': Sigma}, 
            {'num_modes': U_fine.shape[1]})

# 6. Visualization of results (optional)
# If your domain is n_fine x n_fine, compute n_fine from the dimension of U_fine.
# Remember there are 2 fields (u and v). If the shape of U_fine is (2 * n_fine^2, r),
# then n_fine = int(np.sqrt(U_fine.shape[0] // 2)).
n_fine = int(np.sqrt(U_fine.shape[0] // 2))
# plot_2d_system_pod_modes(U_fine, train_data['x'], train_data['y'], n_fine)
# plot_variance(Sigma)
# plot_eigenvalues(Sigma)
###############################################################################
# 8. Down-sample the fine-grid modes to a coarser grid
###############################################################################
# Suppose you have a coarse grid n_coarse. For example: 64, 32, or 16, etc.
n_coarse_list = [64, 32, 16]  # or any set of target coarse grids

# for n_coarse in n_coarse_list:
#     if n_fine % n_coarse != 0:
#         logging.warning(f"Skipping n_coarse={n_coarse} because n_fine={n_fine} is not divisible.")
#         continue
    
#     logging.info(f"Down-sampling POD modes from {n_fine}x{n_fine} to {n_coarse}x{n_coarse}.")
#     U_coarse = downsample_2d_system_pod_modes(U_fine, n_fine, n_coarse)
#     x2 = np.linspace(-10, 10, n_coarse + 1)
#     x = x2[:-1]
#     y = x
#     # Now U_coarse has shape (2 * n_coarse^2, r).
#     # You could also save these coarse modes or do further analysis:
#     coarse_output_name = f'pod_basis_system_{resolution}_{n_coarse}x{n_coarse}.h5'
#     save_hdf5(
#         coarse_output_name,
#         {'POD_modes_coarse': U_coarse,
#         'singular_values_fine': Sigma},{  # same Sigma from the fine SVD
#         'num_modes': U_coarse.shape[1]}
#     )

#     logging.info(f"Saved down-sampled coarse POD basis to {coarse_output_name}")

# logging.info("POD processing (fine + coarse) pipeline completed successfully.")
