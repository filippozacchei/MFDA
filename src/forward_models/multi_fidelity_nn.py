from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Add, Input, Concatenate
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l2

from typing import List, Dict, Tuple, Any
from single_fidelity_nn import *

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

class MultiFidelityNN(SingleFidelityNN):
    """
    A class representing a multi-fidelity neural network model that inherits from SingleFidelityNN.
    Handles multiple fidelity inputs and merges them for the final prediction.

    Attributes:
        input_shapes: List[Tuple[int]] - List of shapes of the inputs for each fidelity level.
        merge_mode: str - Method to merge the outputs of different fidelity levels (e.g., 'add', 'concat').
    """

    def __init__(self,
                 input_shapes: List[Tuple[int]],
                 coeff: float,
                 layers_config: List[Dict[str, Any]],
                 train_config: Dict[str, Any],
                 output_units: int,
                 output_activation: str,
                 merge_mode: str = 'add',
                 correction = False,
                 submodel = None):
        """
        Initialize the MultiFidelityNN model with the given configuration.

        :param input_shapes: List of input shapes for each fidelity level.
        :param coeff: L2 regularization coefficient.
        :param layers_config: List of configurations for each hidden layer (units, activation).
        :param train_config: Training configuration (epochs, batch size, KFold splits).
        :param output_units: Number of units in the output layer.
        :param output_activation: Activation function for the output layer.
        :param merge_mode: How to merge the outputs from different fidelities ('add', 'concat').
        """
        self.input_shapes = input_shapes
        self.merge_mode = merge_mode
        self.additive_correction = correction

        if self.additive_correction is True:
            self.submodel = submodel

        super().__init__(input_shape=input_shapes[0],  # Assume the first fidelity as the main input shape
                         coeff=coeff,
                         layers_config=layers_config,
                         train_config=train_config,
                         output_units=output_units,
                         output_activation=output_activation)

    def build_model(self) -> Model:
        """
        Build and compile a multi-fidelity neural network model with multiple inputs.

        :return: A compiled Keras Model with multiple fidelity inputs.
        """
        # Input layers for each fidelity level
        inputs = [Input(shape=shape) for shape in self.input_shapes]
        fidelities =  [item for item in self.layers_config['input_layers']]

        # Hidden layers for each fidelity level
        fidelity_layers = [self._build_hidden_layers(input_layer,fidelity) for (input_layer,fidelity) in zip(inputs,fidelities)]

        # Merge fidelity layers based on the merge_mode
        if self.merge_mode == 'add':
            merged_output = Add()(fidelity_layers)
        elif self.merge_mode == 'concat':
            merged_output = Concatenate()(fidelity_layers)
        else:
            raise ValueError(f"Unsupported merge_mode: {self.merge_mode}. Use 'add' or 'concat'.")
        
        for layer in self.layers_config['output_layers'][1:]:
            merged_output = Dense(layer['units'], activation=layer['activation'], kernel_regularizer=l2(self.coeff))(merged_output)

        # Output layer
        output = Dense(self.output_units, activation=self.output_activation, kernel_regularizer=l2(self.coeff))(merged_output)

        if self.additive_correction is True:
            model_nn = load_model(self.submodel)
            model_nn.trainable = False
            output_lf = model_nn(inputs[0])
            output = Add()([output,output_lf])

        # Create and compile the model
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=self.train_config.get('optimizer', 'adam'), 
                      loss='mean_squared_error', 
                      metrics=['mean_squared_error'])

        return model

    def _build_hidden_layers(self, input_layer, fidelity):
        """
        Build hidden layers for each input fidelity level.

        :param input_layer: Input layer for a fidelity level.
        :return: Final hidden layer for the corresponding fidelity level.
        """
        x = Dense(self.layers_config['input_layers'][fidelity][0]['units'], activation=self.layers_config['input_layers'][fidelity][0]['activation'], kernel_regularizer=l2(self.coeff))(input_layer)
        for layer in self.layers_config['input_layers'][fidelity][1:]:
            x = Dense(layer['units'], activation=layer['activation'], kernel_regularizer=l2(self.coeff))(x)
        return x

    def kfold_train(self, X_train_fidelities: List[np.ndarray], y_train: np.ndarray) -> None:
        """
        Train the multi-fidelity neural network model using K-Fold cross-validation.

        :param X_train_fidelities: List of training inputs for each fidelity level.
        :param y_train: Target training data.
        """
        kf = KFold(n_splits=self.train_config['n_splits'], shuffle=True, random_state=56)
        fold_var = 1

        for train_index, val_index in kf.split(y_train):
            print(f"Training fold {fold_var}...")

            # Split data into training and validation sets for each fidelity level
            X_train_k_fidelities = [X_train[train_index] for X_train in X_train_fidelities]
            X_val_k_fidelities = [X_train[val_index] for X_train in X_train_fidelities]
            y_train_k, y_val_k = y_train[train_index], y_train[val_index]

            # Build the model
            self.model = self.build_model()

            # Compile the model
            self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

            # Train the model
            self.model.fit(X_train_k_fidelities, y_train_k,
                           epochs=self.train_config['epochs'],
                           batch_size=self.train_config['batch_size'],
                           validation_data=(X_val_k_fidelities, y_val_k),
                           validation_freq = 10,
                           callbacks=[LearningRateScheduler(self.lr_scheduler), PrintEveryNEpoch(10, self.train_config['epochs'])],
                           verbose=0)

            # Save the model for each fold
            model_save_path = os.path.join(self.train_config['model_save_path'], f'model_fold_{fold_var}.keras')
            try:
                os.makedirs(self.train_config['model_save_path'], exist_ok=True)
                self.model.save(model_save_path)
                print(f"Model for fold {fold_var} saved at {model_save_path}")
            except Exception as e:
                print(f"Error saving model for fold {fold_var}: {e}")

            fold_var += 1