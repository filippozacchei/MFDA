import sys
import os
import json
import logging
import tensorflow as tf
import numpy as np
# Add the 'models' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/forward_models/'))

from multi_fidelity_nn import MultiFidelityNN
from utils import *

# Set up logging for progress output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data(config):
    """
    Prepare the training and testing datasets.
    :param config: Configuration dictionary.
    :return: X_train, y_train, X_test, y_test arrays.
    """
    logging.info("Preparing data for multi-fidelity model")
    X_train_param, X_train_coarse, X_train_nn, y_train = prepare_data_multi_fidelity(
        config['n_sample'], 
        config["y_train"],
        config["X_train_param"], 
        config["X_train_coarse"], 
        config["X_train_nn"] 
        )
    X_test_param, X_test_coarse, X_test_nn, y_test = prepare_data_multi_fidelity(
        config['n_sample'],
        config["y_test"],
        config["X_test_param"], 
        config["X_test_coarse"], 
        config["X_test_nn"] 
    )
    return X_train_param, X_train_coarse, X_train_nn, y_train, X_test_param, X_test_coarse, X_test_nn, y_test

def main():         
    # Add the 'models' directory to the Python path
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/forward_models/'))

    # Load configuration
    config_filepath = 'config_MultiFidelity_2step.json'
    config = load_config(config_filepath)

    destination_folder = config["train_config"]["model_save_path"]
    shutil.copy(config_filepath, destination_folder)

    # Prepare training and testing data
    X_train_param, X_train_coarse, X_train_nn, y_train, X_test_param, X_test_coarse, X_test_nn, y_test = prepare_data(config)

    # Log data shapes for debugging
    logging.info(f"Training data shape: X_train: {X_train_param.shape}, y_train: {y_train.shape}")
    logging.info(f"Test data shape: X_test: {X_test_param.shape}, y_test: {y_test.shape}")

    # Initialize the multiFidelityNN model
    logging.info("Correctness of Multi Fidelity Data")
    print(y_test)
    logging.info(f"\nMSE:  {np.sqrt(np.mean((X_test_coarse - y_test)**2)):.4e}")
    print(X_test_coarse)
    logging.info(f"\nMSE:  {np.sqrt(np.mean((X_test_nn - y_test)**2)):.4e}")
    print(X_test_nn)
    
    # Initialize the multiFidelityNN model
    logging.info("Initializing the MultiFidelityNN model")
    mfnn_model = MultiFidelityNN(
        input_shapes=[(X_train_param.shape[1],),
                      (X_train_coarse.shape[1],),
                      (X_train_nn.shape[1],)],
        merge_mode=config["merge_mode"],
        coeff=config["coeff"],
        layers_config=config["layers_config"],
        train_config=config["train_config"],
        output_units=y_train.shape[1],
        output_activation=config["output_activation"]
    )

    # Build the model
    logging.info("Building the multiFidelityNN model")
    mfnn_model.build_model()

    # Train the model using K-Fold cross-validation
    logging.info("Starting K-Fold training")
    mfnn_model.kfold_train([X_train_param,X_train_coarse,X_train_nn], y_train)

    # plot_results(X_test, y_test, mfnn_model.model)

    logging.info("Training completed successfully!")

if __name__ == "__main__":
    main()