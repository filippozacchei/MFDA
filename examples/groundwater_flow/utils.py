import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import sys
import shutil

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


def plot_results(X_test, y_test, model):
    x_data = y_data = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    X,Y = np.meshgrid(x_data, y_data)

    samples = [ 1, 10, 100]
    #Plot POD coefficients: LF vs HF
    fig = plt.figure(figsize=(12,12))
    plt.subplots_adjust(hspace=0.5)
    title = 'True vs Reconstucted signals with NN'
    fig.suptitle(title, fontsize=14)

    for mode in range(3):
        ax = fig.add_subplot(331 + mode)
        pcm = plt.pcolormesh(X, Y, y_test[mode, :].reshape((5, 5)))
        ax.title.set_text('True signal sample: ' + str(samples[mode]))
        plt.colorbar(pcm, ax=ax)
        
        ax = fig.add_subplot(331 + mode + 3)
        reconstructed_sample = np.array(model(X_test[mode, :].reshape((1, 64)))).reshape((5, 5))
        err = y_test[mode, :].reshape(5, 5) - reconstructed_sample
        pcm = plt.pcolormesh(X, Y, reconstructed_sample.reshape((5, 5)))
        ax.title.set_text('NN prediction sample: ' + str(samples[mode]))
        plt.colorbar(pcm, ax=ax)
        
        ax = fig.add_subplot(331 + mode + 6)
        pcm = plt.pcolormesh(X, Y, err)
        ax.title.set_text('Reconst. error sample: ' + str(samples[mode]))
        plt.colorbar(pcm, ax=ax)
    
    plt.show()