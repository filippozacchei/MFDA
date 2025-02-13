import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import sys
import h5py
import shutil

def load_hdf5(filename: str):
    """
    Loads an HDF5 dataset and returns all datasets dynamically.
    :param filename: Path to the HDF5 file.
    :return: Dictionary of dataset names mapped to their values.
    """
    try:
        logging.info(f"Loading HDF5 file: {filename}")
        with h5py.File(filename, 'r') as h5file:
            return {key: h5file[key][:] for key in h5file.keys()}
    except FileNotFoundError:
        logging.error(f"HDF5 file not found: {filename}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading HDF5 file {filename}: {e}")
        sys.exit(1)
        
def save_hdf5(filename, datasets, metadata=None):
    """Saves datasets to an HDF5 file."""
    with h5py.File(filename, 'w') as h5file:
        for key, data in datasets.items():
            h5file.create_dataset(key, data=data)
        if metadata:
            for key, value in metadata.items():
                h5file.attrs[key] = value
                
def export_csv(filename, dataframe):
    """Exports a Pandas DataFrame to CSV."""
    dataframe.to_csv(filename, index=False)
    
def load_config(config_filepath):
    """
    Load configuration from a JSON file.
    :param config_filepath: Path to the configuration file.
    :return: Configuration dictionary.
    """
    try:
        logging.info(f"Loading configuration from {config_filepath}")
        with open(config_filepath, 'r') as config_file:
            config = json.load(config_file)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_filepath}")
        sys.exit(1)

    return config

def load_data(filepath):
    """
    Load data from a CSV file.
    """
    return np.loadtxt(filepath, delimiter=',')

def prepare_data_single_fidelity(n_sample, X_train_filepath, y_train_filepath, X_test_filepath, y_test_filepath):
    """
    Load and prepare data for single-fidelity model.
    """
    X_train = load_data(X_train_filepath)[:n_sample]
    y_train = load_data(y_train_filepath)[:n_sample]
    X_test = load_data(X_test_filepath)
    y_test = load_data(y_test_filepath)
    return X_train, y_train, X_test, y_test

def prepare_data_multi_fidelity(n_sample, y_train_filepath, *input_filepaths):
    """
    Load and prepare data for multi-fidelity models with a variable number of inputs.
    
    :param n_sample: Number of samples to load
    :param y_train_filepath: Filepath for the high-fidelity output data
    :param input_filepaths: Variable number of filepaths for the fidelity levels input data
    :return: A tuple with all input arrays followed by the output array
    """
    # Load the high-fidelity output (y_train)
    y_train = load_data(y_train_filepath)[:n_sample]
    
    # Load all input files for the different fidelities
    X_train_inputs = [load_data(filepath)[:n_sample] for filepath in input_filepaths]
    
    # Return the input arrays followed by the output array
    return (*X_train_inputs, y_train)

def prepare_lstm_dataset(param_values, time_values, time_coefficients):
    """
    Prepares dataset for LSTM where input consists of user-defined parameters, time, and coarse modes at time t.
    
    :param param_values: Array of parameter values with shape (num_samples, num_params).
    :param time_values: Array of time values with shape (time_steps,).
    :param time_coefficients: Time coefficients, shape (num_samples, time_steps, num_modes).
    :return: Numpy array X with shape (num_samples, time_steps, num_params + 1 + num_modes).
    """
    logging.info("Preparing LSTM dataset.")
    
    num_samples, time_steps, _ = time_coefficients.shape
    
    # Expand parameters and time values
    param_expanded = np.repeat(param_values[:, np.newaxis, :], time_steps, axis=1)  # (num_samples, time_steps, num_params)
    time_expanded = np.tile(time_values, (num_samples, 1))[:, :, np.newaxis]  # (num_samples, time_steps, 1)
    
    # Concatenate along the feature dimension
    X = np.concatenate([param_expanded, time_expanded, time_coefficients], axis=2)  # (num_samples, time_steps, num_params + 1 + num_modes)
    
    logging.info(f"LSTM dataset prepared with shape {X.shape}.")
    return X
