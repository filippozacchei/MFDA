import os
import sys
import json
import logging
import shutil
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import h5py

# Add required paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.extend([os.path.join(BASE_DIR, 'forward_models'), os.path.join(BASE_DIR, 'utils')])

# Import necessary modules
from single_fidelity_lstm import SingleFidelityLSTM
from pod_utils import reshape_to_pod_2d_system_snapshots, project, reshape_to_lstm, reconstruct, reconstruct_eff
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
    logging.info("PrepariGng additional input features.")
    X_train_init = np.column_stack((train_data["d1"], train_data["beta"]))
    X_test_init = np.column_stack((test_data["d1"], test_data["beta"]))

    # Generate LSTM-ready datasets
    X_train_prep = prepare_lstm_dataset(X_train_init, train_data["t"], v_train_lstm)
    X_test_prep = prepare_lstm_dataset(X_test_init, test_data["t"], v_test_lstm)

    # Split into features and targets
    X_train, y_train = X_train_prep[:, :, :3], X_train_prep[:, :, 3:]*Sigma[:40]
    X_test, y_test = X_test_prep[:, :, :3], X_test_prep[:, :, 3:]*Sigma[:40]
    
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    #y_train = scaler_Y.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
    #y_test = scaler_Y.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)

    return X_train, y_train, X_test, y_test, U, Sigma, u_test_snapshots, scaler_Y


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
    
    sfnn_model = SingleFidelityLSTM(
        input_shape=(X_train.shape[1], X_train.shape[2]),  # (time_steps, features)
        coeff=config["coeff"],
        layers_config=config["layers_config"],
        train_config=config["train_config"],
        output_units=y_train.shape[2],  # Ensure output is 3D
        output_activation=config["output_activation"]
    )

    logging.info("Building and training the SingleFidelityNN model.")
    sfnn_model.build_model()
    sfnn_model.train(X_train, y_train, X_test, y_test)

    logging.info("Model training completed successfully!")


def evaluate_model(config, X_test, U, Sigma, u_test_snapshots, scaler_Y):
    """Loads the trained model and evaluates reconstruction performance."""
    destination_folder = config["train_config"]["model_save_path"]
    model_path = os.path.join(destination_folder, 'model.keras')

    logging.info("Loading trained model from %s", model_path)
    model = tf.keras.models.load_model(model_path)
 
    predictions = model.predict(X_test)/Sigma[:40]
    #predictions = scaler_Y.inverse_transform(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape)
    predictions_reshaped = predictions.reshape(-1, predictions.shape[-1])

    logging.info("Performing POD-based reconstruction...")
    reconstructed, error = reconstruct(U, Sigma, predictions_reshaped, num_modes=40, original_snapshots=u_test_snapshots)

    logging.info(f"RMSE Test Error: {error:.6f}")

    return model

def main():
    """Main execution function."""
    config_filepath = 'config/config_SingleFidelity.json'
    config = load_configuration(config_filepath)
    train_data = load_hdf5(config["train"])
    test_data = load_hdf5(config["test"])
    
    # Backup configuration
    destination_folder = config["train_config"]["model_save_path"]
    shutil.copy(config_filepath, destination_folder)

    # Prepare datasets
    X_train, y_train, X_test, y_test, U, Sigma, u_test_snapshots, scaler_Y = load_and_process_data(config, num_modes=40)
    # Train the model
    train_model(config, X_train, y_train, X_test, y_test, Sigma)

    # Evaluate the model
    model = evaluate_model(config, X_test, U, Sigma, u_test_snapshots, scaler_Y)
    
    prediction = model.predict(X_train)/Sigma[:40]
    #prediction = scaler_Y.inverse_transform(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape)
    prediction=prediction.reshape(-1, prediction.shape[-1])
    u_train,v_train = reconstruct_eff(U,Sigma,prediction,num_modes=40,nt=101,n_sample=200)
    output_name = 'reaction_diffusion_training_h4_nn.h5' 
    num_samples=200
    t = train_data['t']
    n = 16
    print(u_train.shape)
    with h5py.File(output_name, 'w') as h5file:
        # Create datasets for parameters and solutions
        d1_dset = h5file.create_dataset("d1", (num_samples,), dtype='float64')
        beta_dset = h5file.create_dataset("beta", (num_samples,), dtype='float64')
        u_dset = h5file.create_dataset("u", (num_samples, len(t), n, n), dtype='float64')
        v_dset = h5file.create_dataset("v", (num_samples, len(t), n, n), dtype='float64')
        t_dset = h5file.create_dataset("t", (len(t),), dtype='float64')
        x_dset = h5file.create_dataset("x", (n,), dtype='float64')
        y_dset = h5file.create_dataset("y", (n,), dtype='float64')

        # Store grid and time values (same for all samples)
        t_dset[:] = train_data["t"]
        x_dset[:] = train_data["x"]
        y_dset[:] = train_data["y"]

        # Populate the dataset for each sample
        for i, data in enumerate(train_data):
            d1_dset[i] = train_data['d1'][i]
            beta_dset[i] = train_data['beta'][i]
            u_dset[i] = np.transpose(u_train[i,:,:], (2, 0, 1))  # Transpose (128, 128, 401) -> (401, 128, 128)
            v_dset[i] = np.transpose(v_train[i,:,:], (2, 0, 1))  # Transpose (128, 128, 401) -> (401, 128, 128) 

    prediction = model.predict(X_test)/Sigma[:40]
    #prediction = scaler_Y.inverse_transform(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape)
    prediction= prediction.reshape(-1, prediction.shape[-1])
    u_test, v_test = reconstruct_eff(U,Sigma,prediction,num_modes=40,nt=101,n_sample=20)
    output_name = 'reaction_diffusion_testing_h4_nn.h5' 
    num_samples=20
    with h5py.File(output_name, 'w') as h5file:
        # Create datasets for parameters and solutions
        d1_dset = h5file.create_dataset("d1", (num_samples,), dtype='float64')
        beta_dset = h5file.create_dataset("beta", (num_samples,), dtype='float64')
        u_dset = h5file.create_dataset("u", (num_samples, len(t), n, n), dtype='float64')
        v_dset = h5file.create_dataset("v", (num_samples, len(t), n, n), dtype='float64')
        t_dset = h5file.create_dataset("t", (len(t),), dtype='float64')
        x_dset = h5file.create_dataset("x", (n,), dtype='float64')
        y_dset = h5file.create_dataset("y", (n,), dtype='float64')

        # Store grid and time values (same for all samples)
        t_dset[:] = test_data["t"]
        x_dset[:] = test_data["x"]
        y_dset[:] = test_data["y"]

        # Populate the dataset for each sample
        for i, data in enumerate(test_data):
            d1_dset[i] = test_data['d1'][i]
            beta_dset[i] = test_data['beta'][i]
            u_dset[i] = np.transpose(u_test[i,:,:], (2, 0, 1))  # Transpose (128, 128, 401) -> (401, 128, 128)
            v_dset[i] = np.transpose(v_test[i,:,:], (2, 0, 1))  # Transpose (128, 128, 401) -> (401, 128, 128)

    n = int(np.sqrt(prediction.shape[0]/2))

    predictions = model.predict(X_test)/Sigma[:40] # Shape: (batch_size, time_steps, num_modes)
    #predictions = scaler_Y.inverse_transform(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape)
    predictions_reshaped = predictions.reshape(-1, predictions.shape[-1])
    prediction = reconstruct(U,Sigma,predictions_reshaped,num_modes=40)
    n = int(np.sqrt(prediction.shape[0]//2))
    nt=100
    plot_2d_system_prediction(u_test_snapshots[:,nt:], train_data['x'], train_data['y'], n, n_steps=1010, save_path='./gif/exact_sf1.gif')
    plot_2d_system_prediction(prediction[:,nt:], train_data['x'], train_data['y'], n, n_steps=1010, save_path='./gif/predicted_sf1.gif')
    plot_2d_system_prediction(np.log(np.abs(u_test_snapshots[:,nt:]-prediction[:,nt:])), train_data['x'], train_data['y'], n, n_steps=1010, save_path='./gif/difference_sf1.gif')

if __name__ == "__main__":
    main()
