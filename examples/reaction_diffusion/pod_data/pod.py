import sys
import os
import json
import logging
import tensorflow as tf
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the 'models' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src/utils/'))

from data_utils import load_hdf5, save_hdf5
from pod_utils import *
from plot_utils import *


resolution = 'h1'  
    
def get_output_name(resolution=''):
    """Generates the output filename for POD basis."""
    return f'pod_basis_system_{resolution}.h5'

def get_data_filenames(data_path, resolution=''):
    """Generates training and testing dataset filenames based on resolution."""
    return (
        os.path.join(data_path, f'reaction_diffusion_training_{resolution}.h5'),
        os.path.join(data_path, f'reaction_diffusion_testing_{resolution}.h5')
    )

if __name__ == "__main__":
    logging.info("Starting POD processing pipeline.")
    
    data_path = "../data/"
    
    # Generate output filename
    output_name = get_output_name(resolution)
    
    # Load datasets
    logging.info("Loading training and testing datasets.")
    train_filename, test_filename = get_data_filenames(data_path, resolution)
    train_data = load_hdf5(train_filename)
    test_data = load_hdf5(test_filename)
    
    # Reshape data into 2D snapshots
    logging.info("Reshaping training and testing data.")
    u_train_snapshots = reshape_to_pod_2d_system_snapshots(train_data['u'][:20],train_data['v'][:20])
    u_test_snapshots = reshape_to_pod_2d_system_snapshots(test_data['u'][:20],test_data['v'][:20])
    print(u_train_snapshots.shape)
    # Compute POD using randomized SVD
    logging.info("Computing POD basis using randomized SVD.")
    U, Sigma, VT = randomized_svd(u_train_snapshots, n_components=100)
    VT_test = project(U, Sigma, u_test_snapshots, num_modes=100)
    
    # Reconstruction and error computation for training data
    logging.info("Reconstructing training data and computing error.")
    _, error_rmse_train = reconstruct(U, Sigma, VT, 100, u_train_snapshots)
    logging.info(f"Train Reconstruction RMSE Error: {error_rmse_train:.12f}")
    
    # Reconstruction and error computation for testing data
    logging.info("Reconstructing testing data and computing error.")
    _, error_rmse_test = reconstruct(U, Sigma, VT_test, 100, u_test_snapshots)
    logging.info(f"Test Reconstruction RMSE Error: {error_rmse_test:.12f}")
    
    # Save computed POD basis
    logging.info("Saving POD basis and singular values to HDF5 file.")
    save_hdf5(output_name, {'POD_modes': U, 'singular_values': Sigma}, {'num_modes': U.shape[1]})
    
    # Visualization of results
    # logging.info("Generating POD mode visualizations.")
    plot_2d_system_pod_modes(U, train_data['x'], train_data['y'], int(np.sqrt(U.shape[0]/2)))
    plot_variance(Sigma)
    plot_eigenvalues(Sigma)
    
    # Truncated reconstruction (using only 21 modes)
    logging.info("Performing truncated reconstruction using 21 modes.")
    for i in range(100):
        _, error_rmse_test = reconstruct(U, Sigma, VT_test, i, u_test_snapshots)
        logging.info(f"Test truncated Reconstruction RMSE Error for {i} modes: {error_rmse_test:.12f}")
    
    logging.info("POD processing pipeline completed successfully.")
