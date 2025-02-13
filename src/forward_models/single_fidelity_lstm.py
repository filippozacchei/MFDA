import os
import numpy as np
from typing import List, Dict, Tuple, Any, Callable
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
from tensorflow.keras.optimizers import Adam

import sys

class PrintEveryNEpoch(Callback):
    def __init__(self, n, total_epochs):
        super(PrintEveryNEpoch, self).__init__()
        self.n = n
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n == 0 or (epoch + 1) == self.total_epochs:  # Print every `n` epochs
            logs = logs or {}
            # Format logs, display small values in scientific notation
            def format_value(value):
                return f'{value:.4f}' if value >= 1e-3 else f'{value:.4e}'

            log_str = ' - '.join([f'{key}: {format_value(value)}' for key, value in logs.items()])
            
            # Calculate progress percentage
            progress = (epoch + 1) / self.total_epochs * 100
            progress_bar = f"[{'=' * (int(progress) // 2)}{' ' * (50 - int(progress) // 2)}]"

            # Print progress and logs, overwrite the previous line
            sys.stdout.write(f'\rEpoch {epoch + 1}/{self.total_epochs} | {log_str} | {progress_bar} {progress:.2f}%')
            sys.stdout.flush()

    def on_train_end(self, logs=None):
        print("\nTraining complete.")
            

def make_scheduler(coeff: float, mode: str = 'linear', step=10) -> Callable[[int, float], float]:
    """
    Creates a learning rate scheduler function based on the mode and coefficient.

    :param coeff: The coefficient to modify the learning rate.
    :param mode: The mode of the learning rate schedule (e.g., 'linear').
    :return: A scheduler function to adjust the learning rate over epochs.
    """
    if mode == 'linear':
        def scheduler(epoch: int, lr: float) -> float:
            if epoch < step:
                return lr
            else:
                return max(lr * coeff, 1e-7)
        return scheduler
    elif mode == 'decay':
        def scheduler(epoch: int, lr: float) -> float:
            if epoch < step:
                return lr
            else:
                return lr*(1+coeff*epoch)/(1+coeff*(epoch+1))
        return scheduler

    else:
        raise ValueError(f"Unsupported mode: {mode}. Currently, only 'linear' is supported.")

from single_fidelity_nn import *
class SingleFidelityLSTM(SingleFidelityNN):
    """
    A class representing a single-fidelity neural network model.

    Attributes:
        input_shape: Tuple[int] - Shape of the input data.
        coeff: float - L2 regularization coefficient.
        layers_config: List[Dict[str, Any]] - Configuration of the neural network layers.
        train_config: Dict[str, Any] - Configuration for training the model.
        output_units: int - Number of units in the output layer.
        output_activation: str - Activation function for the output layer.
        model: Sequential - The built Keras model.
        lr_scheduler: Callable - Learning rate scheduler.
    """

    def __init__(self,
                 input_shape: Tuple[int],
                 coeff: float,
                 layers_config: List[Dict[str, Any]],
                 train_config: Dict[str, Any],
                 output_units: int,
                 output_activation: str,
                 rate: int = 0.2,
                 custom_loss = None):                
        """
        Initialize the SingleFidelityNN model with the given configuration.

        :param input_shape: Shape of the input data.
        :param coeff: L2 regularization coefficient.
        :param layers_config: List of configurations for each hidden layer (units, activation).
        :param train_config: Training configuration (epochs, batch size, KFold splits).
        :param output_units: Number of units in the output layer.
        :param output_activation: Activation function for the output layer.
        """
        super().__init__(input_shape,
                       coeff,
                       layers_config,
                       train_config, 
                       output_units, 
                       output_activation, 
                       rate, 
                       custom_loss)

    def build_model(self) -> Sequential:
        """
        Build and compile a single-fidelity neural network model based on the configuration.

        :return: A compiled Keras Sequential model.
        """
        
        model = Sequential()
        
        first_layer = self.layers_config[0]
        
        model.add(Input(shape=self.input_shape))
        
        # Add hidden layers
        for layer in self.layers_config:
            if layer["type"] == "Dense":
                model.add(Dense(layer['units'], 
                        activation=layer['activation'], 
                        kernel_regularizer=l2(self.coeff), kernel_initializer='glorot_uniform'))
            elif layer["type"] == "LSTM":
                model.add(LSTM(layer['units'], return_sequences=layer['return_seq'],
                        activation=layer['activation'],
                        kernel_regularizer=l2(self.coeff)))
            model.add(Dropout(rate=layer['rate']))
            
        # Add output layer
        model.add(Dense(self.output_units, activation=self.output_activation, kernel_regularizer=l2(self.coeff)))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss=self.custom_loss if self.custom_loss is not None else 'mean_squared_error', 
                      metrics=['mean_squared_error', self.custom_loss])
        return model

    def kfold_train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the neural network model using K-Fold cross-validation.

        :param X_train: Input training data.
        :param y_train: Target training data.
        """
        kf = KFold(n_splits=self.train_config['n_splits'], shuffle=True, random_state=42)
        fold_var = 1

        for train_index, val_index in kf.split(X_train):
            print(f"Training fold {fold_var}...")

            X_train_k, X_val_k = X_train[train_index], X_train[val_index]
            y_train_k, y_val_k = y_train[train_index], y_train[val_index]

            # Compile the model
            self.model = self.build_model()

            # Train the model
            self.model.fit(X_train_k, y_train_k,
                           epochs=self.train_config['epochs'],
                           batch_size=self.train_config['batch_size'],
                           validation_data=(X_val_k, y_val_k),
                           validation_freq = 10,
                           callbacks=[LearningRateScheduler(self.lr_scheduler), PrintEveryNEpoch(10, self.train_config['epochs'])],
                           verbose=0)

            # Save the model for each fold
            model_save_path = os.path.join(self.train_config['model_save_path'], f'model_fold_{fold_var}.keras')

            try:
                self.model.save(model_save_path)
                print(f"Model for fold {fold_var} saved at {model_save_path}")
            except Exception as e:
                print(f"Error saving model for fold {fold_var}: {e}")

            fold_var += 1
        return
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None,  y_val: np.ndarray = None) -> None:
        """
        Train the neural network model using K-Fold cross-validation.

        :param X_train: Input training data.
        :param y_train: Target training data.
        """
        
        self.model = self.build_model()

        # Train the model
        self.model.fit(X_train, y_train,
                        epochs=self.train_config['epochs'],
                        batch_size=self.train_config['batch_size'],
                        validation_data=(X_val, y_val),
                        validation_freq = 10,
                        callbacks=[LearningRateScheduler(self.lr_scheduler), PrintEveryNEpoch(10, self.train_config['epochs'])],
                        verbose=0)

        try:
            model_save_path = os.path.join(self.train_config['model_save_path'], f'model.keras')
            self.model.save(model_save_path)
        except Exception as e:
            print(f"Error saving model: {e}")
        return

    def load_model(self, model_path: str) -> None :
        self.model = load_model(model_path)
        return
    

