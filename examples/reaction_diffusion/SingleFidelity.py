import os
import sys
import json
import logging
import shutil
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Add required paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.extend([os.path.join(BASE_DIR, 'forward_models'), os.path.join(BASE_DIR, 'utils')])

# Import necessary modules
from single_fidelity_nn import SingleFidelityNN
from pod_utils import reshape_to_pod_2d_system_snapshots, project, reshape_to_lstm, reconstruct
from data_utils import load_hdf5, prepare_lstm_dataset
from plot_utils import plot_2d_system_prediction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
def load_configuration(config_path):
    """Load JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_and_process_data(config, num_modes=10):
    """
    Load datasets, apply POD projection, and prepare LSTM-ready data.
    :param config: Configuration dictionary.
    :param num_modes: Number of POD modes to retain.
    :return: Processed training and testing data.
    """
    logging.info("Loading datasets...")
    train_data = load_hdf5(config["train"])
    test_data = load_hdf5(config["test"])
    pod_basis = load_hdf5(config["pod_basis"])

    # Reshape data into 2D POD snapshots
    logging.info("Reshaping datasets into 2D POD snapshots.")
    u_train_snapshots = reshape_to_pod_2d_system_snapshots(train_data['u'], train_data['v'])
    u_test_snapshots = reshape_to_pod_2d_system_snapshots(test_data['u'], test_data['v'])

    # Project data onto POD modes
    logging.info("Projecting data onto POD basis with %d modes.", num_modes)
    U, Sigma = pod_basis["POD_modes"], pod_basis["singular_values"]
    v_train = project(U, Sigma, u_train_snapshots, num_modes=num_modes)
    v_test = project(U, Sigma, u_test_snapshots, num_modes=num_modes)

    logging.info(f"Projected training data shape: {v_train.shape}")
    logging.info(f"Projected test data shape: {v_test.shape}")

    # Reshape for LSTM model
    logging.info("Reshaping projected data for LSTM input.")
    v_train_lstm = reshape_to_lstm(v_train, train_data['u'].shape[0], train_data['u'].shape[1], num_modes)
    v_test_lstm = reshape_to_lstm(v_test, test_data['u'].shape[0], test_data['u'].shape[1], num_modes)

    # Prepare additional input features
    logging.info("Preparing additional input features.")
    X_train_init = np.column_stack((train_data["d1"], train_data["beta"]))
    X_test_init = np.column_stack((test_data["d1"], test_data["beta"]))

    # Generate LSTM-ready datasets
    X_train_prep = prepare_lstm_dataset(X_train_init, train_data["t"], v_train_lstm)
    X_test_prep = prepare_lstm_dataset(X_test_init, test_data["t"], v_test_lstm)

    # Split into features and targets
    X_train, y_train = X_train_prep[:, :, :3], X_train_prep[:, :, 3:]
    X_test, y_test = X_test_prep[:, :, :3], X_test_prep[:, :, 3:]

    # Reshape for Dense Neural Network: (num_samples * time_steps, num_features)
    X_train, y_train = reshape_for_dense_nn(X_train, y_train)
    X_test, y_test = reshape_for_dense_nn(X_test, y_test)

    # **Apply feature scaling**
    X_scaler = MinMaxScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    # **Weight output coefficients by singular values**
    y_train = y_scaler.transform(y_train) 
    y_test = y_scaler.transform(y_test)

    # Log final dataset shapes
    logging.info(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    logging.info(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, y_train, X_test, y_test, U, Sigma, u_test_snapshots, X_scaler, y_scaler


def reshape_for_dense_nn(X, y):
    """
    Reshape dataset to be compatible with a Dense Neural Network.
    :param X: Input features (num_samples, time_steps, features).
    :param y: Target values (num_samples, time_steps, output_features).
    :return: Reshaped X and y.
    """
    num_samples, time_steps, num_features = X.shape
    X = X.reshape(num_samples * time_steps, num_features)
    y = y.reshape(num_samples * time_steps, -1)  # Preserve last dimension
    return X, y


def train_model(config, X_train, y_train, X_test, y_test, Sigma):
    """
    Initializes, builds, and trains the Single Fidelity Neural Network model.
    """
    logging.info("Initializing SingleFidelityNN model.")
    
    sfnn_model = SingleFidelityNN(
        input_shape=(X_train.shape[1],),
        coeff=config["coeff"],
        layers_config=config["layers_config"],
        train_config=config["train_config"],
        output_units=y_train.shape[1],
        output_activation=config["output_activation"]
    )

    logging.info("Building and training the SingleFidelityNN model.")
    sfnn_model.build_model()
    sfnn_model.train(X_train, y_train, X_test, y_test)

    logging.info("Model training completed successfully!")



def evaluate_model(config, X_test, U, Sigma, u_test_snapshots, y_scaler):
    """Loads the trained model and evaluates reconstruction performance."""
    destination_folder = config["train_config"]["model_save_path"]
    model_path = os.path.join(destination_folder, 'model.keras')

    logging.info("Loading trained model from %s", model_path)
    model = tf.keras.models.load_model(model_path)

    logging.info("Performing POD-based reconstruction...")
    reconstructed, error = reconstruct(U, Sigma, y_scaler.inverse_transform(model.predict(X_test)), num_modes=14, original_snapshots=u_test_snapshots)

    logging.info(f"RMSE Test Error: {error:.6f}")

    return model

def main():
    """Main execution function."""
    config_filepath = 'config/config_SingleFidelity.json'
    config = load_configuration(config_filepath)
    train_data = load_hdf5(config["train"])
    print(train_data['u'].shape)
    
    # Backup configuration
    destination_folder = config["train_config"]["model_save_path"]
    shutil.copy(config_filepath, destination_folder)

    # Prepare datasets
    X_train, y_train, X_test, y_test, U, Sigma, u_test_snapshots, _, y_scaler = load_and_process_data(config, num_modes=14)
    
    # Train the model
    train_model(config, X_train, y_train, X_test, y_test, Sigma)

    # Evaluate the model
    model = evaluate_model(config, X_test, U, Sigma, u_test_snapshots, y_scaler)
    
    prediction = y_scaler.inverse_transform(model.predict(X_test))
    prediction = reconstruct(U,Sigma,prediction,num_modes=14)
    n = int(np.sqrt(prediction.shape[0]/2))
    print(prediction.shape)
    plot_2d_system_prediction(u_test_snapshots, train_data['x'], train_data['y'], n, 101)
    plot_2d_system_prediction(prediction, train_data['x'], train_data['y'], n, 101)


if __name__ == "__main__":
    main()