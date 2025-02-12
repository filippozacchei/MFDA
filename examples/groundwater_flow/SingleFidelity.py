import sys
import os
import json
import logging
import tensorflow as tf
# Add the 'models' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/forward_models/'))

from single_fidelity_nn import SingleFidelityNN
from utils import *

# Set up logging for progress output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_data(config):
    """
    Prepare the training and testing datasets.
    :param config: Configuration dictionary.
    :return: X_train, y_train, X_test, y_test arrays.
    """
    logging.info("Preparing data for single-fidelity model")
    return prepare_data_single_fidelity(
        n_sample=config['n_sample'], 
        X_train_filepath=config["X_train"], 
        y_train_filepath=config["y_train"], 
        X_test_filepath=config["X_test"], 
        y_test_filepath=config["y_test"]
    )

def main():         
    # Load configuration
    config_filepath = 'config/config_SingleFidelity.json'
    config = load_config(config_filepath)

    destination_folder = config["train_config"]["model_save_path"]
    shutil.copy(config_filepath, destination_folder)
    
    # Prepare training and testing data
    X_train, y_train, X_test, y_test = prepare_data(config)

    # Log data shapes for debugging
    logging.info(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    logging.info(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Initialize the SingleFidelityNN model
    logging.info("Initializing the SingleFidelityNN model")
    sfnn_model = SingleFidelityNN(
        input_shape=(X_train.shape[1],),
        coeff=config["coeff"],
        layers_config=config["layers_config"],
        train_config=config["train_config"],
        output_units=y_train.shape[1],
        output_activation=config["output_activation"]
    )

    # Build the model
    logging.info("Building the SingleFidelityNN model")
    sfnn_model.build_model()

    # Train the model using K-Fold cross-validation
    logging.info("Starting K-Fold training")
    sfnn_model.kfold_train(X_train, y_train)

    # plot_results(X_test, y_test, sfnn_model.model)

    logging.info("Training completed successfully!")

if __name__ == "__main__":
    main()