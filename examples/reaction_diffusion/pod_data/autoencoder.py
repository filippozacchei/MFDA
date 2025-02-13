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
from autoencoder_utils import reshape_to_autoencoder_input, build_autoencoder, train_autoencoder, save_autoencoder_models, reconstruct_autoencoder

resolution = 'h4'  # Define resolution if needed
    
def get_output_name(resolution=''):
    """Generates the output filename for autoencoder model."""
    return f'autoencoder_model_{resolution}.h5'

def get_data_filenames(data_path, resolution=''):
    """Generates training and testing dataset filenames based on resolution."""
    return (
        os.path.join(data_path, f'reaction_diffusion_training_{resolution}.h5'),
        os.path.join(data_path, f'reaction_diffusion_testing_{resolution}.h5')
    )

if __name__ == "__main__":
    logging.info("Starting Autoencoder processing pipeline.")
    
    data_path = "../data/"
    
    # Generate output filename
    output_name = get_output_name(resolution)
    
    # Load datasets
    logging.info("Loading training and testing datasets.")
    train_filename, test_filename = get_data_filenames(data_path, resolution)
    train_data = load_hdf5(train_filename)
    test_data = load_hdf5(test_filename)
    
    # Reshape data into autoencoder input format
    logging.info("Reshaping training and testing data.")
    u_train_snapshots = reshape_to_autoencoder_input(train_data['u'])
    u_test_snapshots = reshape_to_autoencoder_input(test_data['u'])
    
    input_dim = u_train_snapshots.shape[1]
    latent_dim = 32  # Adjust as needed
    
    # Build Autoencoder
    logging.info("Building and training Autoencoder.")
    autoencoder, encoder, decoder = build_autoencoder(input_dim, latent_dim)
    
    # Train Autoencoder
    history = train_autoencoder(autoencoder, u_train_snapshots, x_val=u_test_snapshots, epochs=200, batch_size=128)
    
    # Save trained models
    logging.info("Saving Autoencoder models.")
    save_autoencoder_models(encoder, decoder, resolution)
    
    # Reconstruction and error computation for training data
    logging.info("Reconstructing training data and computing error.")
    reconstructed_train = reconstruct_autoencoder(encoder, decoder, u_train_snapshots)
    error_rmse_train = np.mean(np.abs(u_train_snapshots - reconstructed_train)/np.abs(reconstructed_train))
    logging.info(f"Train Reconstruction RMSE Error: {error_rmse_train:.12f}")
    
    # Reconstruction and error computation for testing data
    logging.info("Reconstructing testing data and computing error.")
    reconstructed_test = reconstruct_autoencoder(encoder, decoder, u_test_snapshots)
    error_rmse_test = np.mean(np.abs(u_test_snapshots - reconstructed_test)/np.abs(reconstructed_test))
    logging.info(f"Test Reconstruction RMSE Error: {error_rmse_test:.12f}")
    
    logging.info("Autoencoder processing pipeline completed successfully.")