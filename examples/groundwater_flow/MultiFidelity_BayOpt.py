import sys
import os
import json
import logging
import tensorflow as tf
import numpy as np
import optuna
from optuna.integration import TFKerasPruningCallback
import multiprocessing

# Add the 'models' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/forward_models/'))

from multi_fidelity_nn import *
from single_fidelity_nn import *
from utils import *

# Set up logging for progress output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_data(config):
    """
    Prepare the training and testing datasets.
    :param config: Configuration dictionary.
    :return: Training and testing datasets (X_train, y_train, X_test, y_test).
    """
    logging.info("Preparing data for the multi-fidelity model...")
    X_train_param, X_train_coarse1, X_train_coarse2, X_train_coarse3, y_train = prepare_data_multi_fidelity(
        config['n_sample'],
        config["y_train"],
        config["X_train_param"],
        config["X_train_coarse1"],
        config["X_train_coarse2"],
        config["X_train_coarse3"]
    )
    X_test_param, X_test_coarse1, X_test_coarse2, X_test_coarse3, y_test = prepare_data_multi_fidelity(
        12800,
        config["y_test"],
        config["X_test_param"],
        config["X_test_coarse1"],
        config["X_test_coarse2"],
        config["X_test_coarse3"]
    )
    return X_train_param, X_train_coarse1, X_train_coarse2, X_train_coarse3, y_train, X_test_param, X_test_coarse1, X_test_coarse2, X_test_coarse3, y_test


def build_layers_config(n_layers1, n_layers2, n_layers3, n_layers4, n_output,neurons_per_layer, activations1, activations2, activations3, dropout_rate1, dropout_rate2, dropout_rate3):
    """
    Build the layer configuration for the Multi-Fidelity Neural Network.
    :param trial: The current Optuna trial.
    :return: Dictionary containing input and output layer configurations.
    """
    return {
        'input_layers': {
            "param": [{"units": 2**neurons_per_layer, "activation": activations1, "rate": dropout_rate1} for _ in range(n_layers1)] +
                     [{"units": 2**neurons_per_layer, "activation": activations2, "rate": dropout_rate2}],
            "coarse_solution1": [{"units": 2**neurons_per_layer, "activation": activations1, "rate": dropout_rate1} for _ in range(n_layers2)] +
                               [{"units": 2**neurons_per_layer, "activation": activations2, "rate": dropout_rate2}],
            "coarse_solution2": [{"units": 2**neurons_per_layer, "activation": activations1, "rate": dropout_rate1} for _ in range(n_layers3)] +
                               [{"units": 2**neurons_per_layer, "activation": activations2, "rate": dropout_rate2}],
            "coarse_solution3": [{"units": 2**neurons_per_layer, "activation": activations1, "rate": dropout_rate1} for _ in range(n_layers4)] +
                               [{"units": 2**neurons_per_layer, "activation": activations2, "rate": dropout_rate2}]
            },
        'output_layers': [{"units": 2**neurons_per_layer, "activation": activations3, "rate": dropout_rate3} for _ in range(n_output)]
    }


def main():
    # Load configuration
    config_filepath = 'config/config_MultiFidelity.json'
    config = load_config(config_filepath)

    # Prepare training and testing data
    X_train_param, X_train_coarse1, X_train_coarse2, X_train_coarse3, y_train, X_test_param, X_test_coarse1, X_test_coarse2, X_test_coarse3, y_test = prepare_data(config)


    def objective(trial):
        # Hyperparameter suggestions
        neurons_per_layer = trial.suggest_int("neurons", 4, 9)
        dropout_rate1 = trial.suggest_uniform("dropout_rate1", 0.0, 0.25)
        dropout_rate2 = trial.suggest_uniform("dropout_rate2", 0.0, 0.25)
        dropout_rate3 = trial.suggest_uniform("dropout_rate3", 0.0, 0.25)
        merge_mode = trial.suggest_categorical("merge_mode", ["add", "concat"])
        activations1 = trial.suggest_categorical("activation1", ["gelu", "tanh", "selu",  "relu", "linear"])
        activations2 = trial.suggest_categorical("activation2", ["gelu", "tanh", "selu",  "relu", "linear"])
        activations3 = trial.suggest_categorical("activation3", ["gelu", "tanh", "selu",  "relu", "linear"])
        n_layers1 = trial.suggest_int("n_layers1", 0, 6)
        n_layers2 = trial.suggest_int("n_layers2", 0, 6)
        n_layers3 = trial.suggest_int("n_layers3", 0, 6)
        n_layers4 = trial.suggest_int("n_layers4", 0, 6)
        n_output = trial.suggest_int("n_output_layers", 0, 6)

        # Build layer configuration
        layers_config = build_layers_config(n_layers1,
                                            n_layers2,
                                            n_layers3,
                                            n_layers4,
                                            n_output,
                                            neurons_per_layer, 
                                            activations1, 
                                            activations2, 
                                            activations3, 
                                            dropout_rate1, 
                                            dropout_rate2, 
                                            dropout_rate3)

        # Training configuration
        train_config = {
            "epochs": 1000,
            "batch_size": 64,
            "n_splits": 1,
            "scheduler_coeff": 0.99,
            "scheduler_mode": "linear",
            "optimizer": "adam",
            "model_save_path": "./models/multi_fidelity/resolution_25-10-5/optuna"
        }

        # Initialize and train the MultiFidelityNN model
        mfnn_model = MultiFidelityNN(
            input_shapes=[(X_train_param.shape[1],), 
                          (X_train_coarse1.shape[1],), 
                          (X_train_coarse2.shape[1],), 
                          (X_train_coarse3.shape[1],)],
            merge_mode=merge_mode,
            coeff=config["coeff"],
            layers_config=layers_config,
            train_config=train_config,
            output_units=y_train.shape[1],
            output_activation=config["output_activation"]
        )

        mfnn_model.model = mfnn_model.build_model()
        callbacks = [TFKerasPruningCallback(trial, "val_loss")]

        history = mfnn_model.model.fit(
            [X_train_param, X_train_coarse1, X_train_coarse2, X_train_coarse3], y_train,
            epochs=train_config['epochs'],
            batch_size=train_config['batch_size'],
            validation_data=([X_test_param, X_test_coarse1, X_test_coarse2, X_test_coarse3], y_test),
            validation_freq=1,
            callbacks=[LearningRateScheduler(mfnn_model.lr_scheduler), PrintEveryNEpoch(50, train_config['epochs'])] + callbacks,
            verbose=0
        )

        # Evaluate model
        predictions = mfnn_model.model([X_test_param, X_test_coarse1, X_test_coarse2, X_test_coarse3])
        mse = np.mean((predictions - y_test) ** 2)
        logging.info(f"Trial {trial.number} | MSE: {mse:.8f} | Params: {trial.params}")
        return mse

    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2000, n_jobs=4)

    logging.info(f"Best Hyperparameters: {study.best_params}")
    logging.info("Training completed successfully!")


if __name__ == "__main__":
    main()