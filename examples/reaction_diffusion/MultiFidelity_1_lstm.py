import os
import sys
import json
import logging
import shutil
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Add required paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.extend([os.path.join(BASE_DIR, 'forward_models'), os.path.join(BASE_DIR, 'utils')])

# Import necessary modules
from multi_fidelity_lstm import MultiFidelityLSTM
from pod_utils import reshape_to_pod_2d_system_snapshots, project, reshape_to_lstm, reconstruct
from data_utils import load_hdf5, prepare_lstm_dataset
from plot_utils import plot_final_U_snapshot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
def load_configuration(config_path):
    """Load JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)
    
def temporal_interpolation_splines(u_data_coarse, time_steps_coarse, time_steps_fine):
    """ Perform temporal interpolation on u_data_coarse using cubic splines
    to match the time dimensionality of the high-fidelity data. """
    num_samples, _, n, n = u_data_coarse.shape
    # Create a normalized time grid for coarse and fine data
    time_coarse = np.linspace(0, 1, time_steps_coarse)
    time_fine = np.linspace(0, 1, time_steps_fine)
    
    # Allocate memory for interpolated data
    u_data_coarse_interpolated = np.zeros((num_samples, time_steps_fine, n, n))
    
    # Perform interpolation for each sample and spatial location
    for sample_idx in range(num_samples):
        for i in range(n):
            for j in range(n):
                # Extract the time series for the current spatial location
                time_series = u_data_coarse[sample_idx, :, i, j]
                # Create a cubic spline interpolator
                spline = interp1d(time_coarse, time_series, kind='cubic', fill_value="extrapolate")
                # Interpolate to the fine time grid
                u_data_coarse_interpolated[sample_idx, :, i, j] = spline(time_fine)
    
    return u_data_coarse_interpolated

def load_and_process_data(config, num_modes=25):
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
    
    train_data_coarse1 = load_hdf5(config["train_coarse1"])
    test_data_coarse1 = load_hdf5(config["test_coarse1"])
    pod_basis_coarse1 = load_hdf5(config["pod_basis_coarse1"])
    
    train_data_coarse1['u'] = temporal_interpolation_splines(
        train_data_coarse1['u'], train_data_coarse1['u'].shape[1], train_data['u'].shape[1]
    )
    
    test_data_coarse1['u'] = temporal_interpolation_splines(
        test_data_coarse1['u'], test_data_coarse1['u'].shape[1], test_data['u'].shape[1]
    )
    
    train_data_coarse1['v'] = temporal_interpolation_splines(
        train_data_coarse1['v'], train_data_coarse1['v'].shape[1], train_data['v'].shape[1]
    )
    
    test_data_coarse1['v'] = temporal_interpolation_splines(
        test_data_coarse1['v'], test_data_coarse1['v'].shape[1], test_data['v'].shape[1]
    )
    
    
    # Reshape data into 2D POD snapshots
    logging.info("Reshaping datasets into 2D POD snapshots.")

    # u_train_snapshots = reshape_to_pod_2d_system_snapshots(train_data['u'], train_data['v'])
    # u_test_snapshots = reshape_to_pod_2d_system_snapshots(test_data['u'], test_data['v'])
    # u_train_snapshots_coarse1 = reshape_to_pod_2d_system_snapshots(train_data_coarse1['u'], train_data_coarse1['v'])
    # u_test_snapshots_coarse1 = reshape_to_pod_2d_system_snapshots(test_data_coarse1['u'], test_data_coarse1['v'])
    
    u_train_snapshots = reshape_to_pod_2d_system_snapshots(train_data['u'],train_data['v'])
    u_test_snapshots = reshape_to_pod_2d_system_snapshots(test_data['u'],test_data['v'])
    
    u_train_snapshots_coarse1 = reshape_to_pod_2d_system_snapshots(train_data_coarse1['u'],train_data_coarse1['v'])
    u_test_snapshots_coarse1 = reshape_to_pod_2d_system_snapshots(test_data_coarse1['u'],test_data_coarse1['v'])


    # Project data onto POD modes
    logging.info("Projecting data onto POD basis with %d modes.", num_modes)
    U, Sigma = pod_basis["POD_modes"], pod_basis["singular_values"]
    v_train = project(U, Sigma, u_train_snapshots, num_modes=num_modes)
    v_test = project(U, Sigma, u_test_snapshots, num_modes=num_modes)
    
    U_coarse1, Sigma_coarse1 = pod_basis_coarse1["POD_modes_coarse"], pod_basis_coarse1["singular_values_fine"]
    v_train_coarse1 = project(U_coarse1, Sigma_coarse1, u_train_snapshots_coarse1, num_modes=num_modes)
    v_test_coarse1 = project(U_coarse1, Sigma_coarse1, u_test_snapshots_coarse1, num_modes=num_modes)

    logging.info(f"Projected training data shape: {v_train.shape}")
    logging.info(f"Projected test data shape: {v_test.shape}")

    # Reshape for LSTM model
    logging.info("Reshaping projected data for LSTM input.")
    v_train_lstm = reshape_to_lstm(v_train, train_data['u'].shape[0], train_data['u'].shape[1], num_modes)
    v_test_lstm = reshape_to_lstm(v_test, test_data['u'].shape[0], test_data['u'].shape[1], num_modes)
    
    v_train_lstm_coarse1 = reshape_to_lstm(v_train_coarse1, train_data['u'].shape[0], train_data['u'].shape[1], num_modes)
    v_test_lstm_coarse1 = reshape_to_lstm(v_test_coarse1, test_data['u'].shape[0], test_data['u'].shape[1], num_modes)

    # Prepare additional input features
    logging.info("Preparing additional input features.")
    X_train_init = np.column_stack((train_data["d1"], train_data["beta"]))
    X_test_init = np.column_stack((test_data["d1"], test_data["beta"]))

    # Generate LSTM-ready datasets
    X_train_prep = prepare_lstm_dataset(X_train_init, train_data["t"], v_train_lstm)
    X_test_prep = prepare_lstm_dataset(X_test_init, test_data["t"], v_test_lstm)
    
    X_train_prep_coarse1 = prepare_lstm_dataset(X_train_init, train_data["t"], v_train_lstm_coarse1)
    X_test_prep_coarse1 = prepare_lstm_dataset(X_test_init, test_data["t"], v_test_lstm_coarse1)

    # Split into features and targets
    X_train, y_train = X_train_prep[:, :, :3], X_train_prep[:, :, 3:]*Sigma[:num_modes]
    X_test, y_test = X_test_prep[:, :, :3], X_test_prep[:, :, 3:]*Sigma[:num_modes]
    
    X_train_coarse1 = X_train_prep_coarse1[:, :, 3:]*Sigma[:num_modes]
    X_test_coarse1 = X_test_prep_coarse1[:, :, 3:]*Sigma[:num_modes]
    
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # y_train = scaler_Y.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
    # y_test = scaler_Y.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
    
    # X_train_coarse1 = scaler_Y.transform(X_train_coarse1.reshape(-1, X_train_coarse1.shape[-1])).reshape(X_train_coarse1.shape)
    # X_test_coarse1 = scaler_Y.transform(X_test_coarse1.reshape(-1, X_test_coarse1.shape[-1])).reshape(X_test_coarse1.shape)

    return X_train, X_train_coarse1, y_train, X_test, X_test_coarse1, y_test, U, Sigma, u_test_snapshots, scaler_Y


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


def train_model(config, X_train, X_train_coarse1, y_train, X_test, X_test_coarse1, y_test, Sigma):
    """
    Initializes, builds, and trains the Single Fidelity Neural Network model.
    """
    logging.info("Initializing SingleFidelityNN model.")
    
    sfnn_model = MultiFidelityLSTM(
        input_shapes=[(X_train.shape[1], X_train.shape[2]), (X_train_coarse1.shape[1], X_train_coarse1.shape[2])], # (time_steps, features)
        coeff=config["coeff"],
        layers_config=config["layers_config"],
        train_config=config["train_config"],
        output_units=y_train.shape[2],  # Ensure output is 3D
        output_activation=config["output_activation"],
        merge_mode=config["merge_mode"]
    )

    logging.info("Building and training the SingleFidelityNN model.")
    sfnn_model.build_model()
    sfnn_model.train([X_train,X_train_coarse1], y_train, X_test_fidelities=[X_test,X_test_coarse1], y_test=y_test)

    logging.info("Model training completed successfully!")



def evaluate_model(config, X_test, X_test_coarse1, U, Sigma, u_test_snapshots, num_modes, scaler_Y):
    """Loads the trained model and evaluates reconstruction performance."""
    destination_folder = config["train_config"]["model_save_path"]
    model_path = os.path.join(destination_folder, 'model.keras')

    logging.info("Loading trained model from %s", model_path)
    model = tf.keras.models.load_model(model_path, safe_mode=False)

    logging.info("Performing POD-based reconstruction...")
    predictions = model.predict((X_test, X_test_coarse1))/Sigma[:num_modes]  # Shape: (batch_size, time_steps, num_modes)
    # predictions = model.predict((X_test, X_test_coarse1))  # Shape: (batch_size, time_steps, num_modes)
    # predictions = scaler_Y.inverse_transform(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape)
    predictions_reshaped = predictions.reshape(-1, predictions.shape[-1])  # Shape: (batch_size * time_steps, num_modes)
    # predictions_inverse = scaler_Y.inverse_transform(predictions_reshaped)

    reconstructed, error = reconstruct(U, Sigma, predictions_reshaped, num_modes, original_snapshots=u_test_snapshots)

    logging.info(f"RMSE Test Error: {error:.6f}")

    return model

def main():
    """Main execution function."""
    config_filepath = 'config/config_MultiFidelity_1.json'
    config = load_configuration(config_filepath)
    train_data = load_hdf5(config["train"])
    
    # Backup configuration
    destination_folder = config["train_config"]["model_save_path"]
    # shutil.copy(config_filepath, destination_folder)
    num_modes=25
    # Prepare datasets
    X_train, X_train_coarse1, y_train, X_test, X_test_coarse1, y_test, U, Sigma, u_test_snapshots, scaler_Y = load_and_process_data(config, num_modes=num_modes)
    print("12211222")
    print(u_test_snapshots)
    # Train the model
    train_model(config, X_train, X_train_coarse1, y_train, X_test, X_test_coarse1, y_test, Sigma)

    # Evaluate the model
    model = evaluate_model(config, X_test, X_test_coarse1, U, Sigma, u_test_snapshots, num_modes=num_modes, scaler_Y=scaler_Y)
    
    predictions = model.predict((X_test, X_test_coarse1))/Sigma[:25] # Shape: (batch_size, time_steps, num_modes)
    #predictions = scaler_Y.inverse_transform(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape)
    predictions_reshaped = predictions.reshape(-1, predictions.shape[-1])  # Shape: (batch_size * time_steps, num_modes)
    # predictions_inverse = scaler_Y.inverse_transform(predictions_reshaped)

    prediction = reconstruct(U,Sigma,predictions_reshaped,num_modes=num_modes)
    n = int(np.sqrt(prediction.shape[0]//2))
    nt=2000
    #plot_2d_system_prediction(u_test_snapshots[:,nt:], train_data['x'], train_data['y'], n, n_steps=10001, save_path='./gif/exact_mf1.gif')
    plot_final_U_snapshot(prediction[:,:nt], train_data['x'], train_data['y'], n, n_steps=1001, save_path='./gif/predicted_mf1.pdf')
    plot_final_U_snapshot(np.abs(u_test_snapshots[:,:nt]-prediction[:,:nt]), train_data['x'], train_data['y'], n, n_steps=1001, save_path='./gif/difference_mf1.pdf',title=r"$|U_{ex}-U_{MF}^{(1)}|$",vmin=0.0,vmax=1.0)

if __name__ == "__main__":
    main()
