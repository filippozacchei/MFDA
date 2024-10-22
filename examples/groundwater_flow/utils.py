import numpy as np

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