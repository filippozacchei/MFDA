import sys
import os
import json
import logging
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

import h5py
# Add the 'models' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/forward_models/'))

from multi_fidelity_nn import MultiFidelityNN
from utils import *

# Set up logging for progress output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data(config):
    """
    Prepare the training and testing datasets.
    :param config: Configuration dictionary.
    :return: X_train, y_train, X_test, y_test arrays.
    """
    logging.info("Preparing data for multi-fidelity model")
    X_train_param, X_train_coarse1, y_train = prepare_data_multi_fidelity(
        config['n_sample'], 
        config["y_train"],
        config["X_train_param"], 
        config["X_train_coarse1"]
        )
    X_test_param, X_test_coarse1, y_test = prepare_data_multi_fidelity(
        12800,
        config["y_test"],
        config["X_test_param"], 
        config["X_test_coarse1"]
    )
    return X_train_param, X_train_coarse1, y_train, X_test_param, X_test_coarse1, y_test

def main():         
    # Add the 'models' directory to the Python path
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/forward_models/'))

    # Load configuration
    config_filepath = 'config/config_MultiFidelity3.json'
    config = load_config(config_filepath)

    destination_folder = config["train_config"]["model_save_path"]
    shutil.copy(config_filepath, destination_folder)

    # Prepare training and testing data
    X_train_param, X_train_coarse1, y_train, X_test_param, X_test_coarse1, y_test = prepare_data(config)
    
    # Load POD basis (modes and singular values)
    with h5py.File('./data/data_generation/pod_data_h1.h5', 'r') as h5file2:
        U_hf= h5file2['POD_modes'][:] # Use first 34 modes
        S_hf = h5file2['singular_values'][:]  # Singular values
        sol_train_hf = h5file2["train_solution"][:config['n_sample']]
        sol_test_hf = h5file2["test_solution"][:]
        
    with h5py.File('./data/data_generation/pod_data_h2.h5', 'r') as h5file3:
        U_lf= h5file3['POD_modes'][:] # Use first 34 modes
        S_lf = h5file3['singular_values'][:]  # Singular values
        sol_train_lf = h5file3["train_solution"][:config['n_sample']]
        sol_test_lf = h5file3["test_solution"][:]
    
    print(sol_train_hf.shape)
    print(U_hf.shape)
    print(U_lf.shape)
    print(S_hf.shape)
    print(S_lf.shape)
     
    n_modes=45
    Uh_hf = U_hf[:,:n_modes] 
    Uh_lf = U_lf[:,:n_modes] 
    Sh_hf = S_hf[:n_modes] 
    Sh_lf = S_lf[:n_modes] 
    Vh_hf = (Uh_hf.T @ sol_train_hf.T) / Sh_hf[:, None]
    Vh_hf_test = (Uh_hf.T @ sol_test_hf.T) / Sh_hf[:, None]
    Vh_lf = (Uh_lf.T @ sol_train_lf.T) / Sh_lf[:, None]
    Vh_lf_test = (Uh_lf.T @ sol_test_lf.T) / Sh_lf[:, None]

    Vh_hf = Vh_hf.T
    Vh_hf_test = Vh_hf_test.T
    Vh_lf = Vh_lf.T
    Vh_lf_test = Vh_lf_test.T
    
    scaler_input = StandardScaler()
    scaler_output = StandardScaler()
    
    Vh_lf_scaled = scaler_input.fit_transform(Vh_lf)    
    Vh_lf_test_scaled = scaler_input.transform(Vh_lf_test)
    
    Vh_hf_scaled = scaler_output.fit_transform(Vh_hf)    
    Vh_hf_test_scaled = scaler_output.transform(Vh_hf_test)
        
    print(Vh_hf.shape)
    
    # X_train_coarse2 = []
    # X_test_coarse2 = []
    # for i in range(1,5):
    #     model_nn = tf.keras.models.load_model("./models/single_fidelity/resolution_h1/samples_8000/model_fold_"+str(i)+".keras")      
    #     third_layer_output = model_nn.layers[4].output  # Adjust index if needed
    #     intermediate_model = tf.keras.Model(inputs=model_nn.input, outputs=third_layer_output)                                
    #     X_train_coarse2.append(intermediate_model(X_train_param[:,:]).numpy())
    #     X_test_coarse2.append(intermediate_model(X_test_param[:,:]).numpy())
        
    # X_train_coarse2=np.mean(np.array(X_train_coarse2),axis=0)
    # X_test_coarse2=np.mean(np.array(X_test_coarse2),axis=0)
    
    # print(X_test_coarse2.shape)

    # Log data shapes for debugging
    logging.info(f"Training data shape: X_train: {X_train_param.shape}, y_train: {y_train.shape}")
    logging.info(f"Test data shape: X_test: {X_test_param.shape}, y_test: {y_test.shape}")

    # Initialize the multiFidelityNN model
    logging.info("Correctness of Multi Fidelity Data")
    
    logging.info(f"\nMSE train:  {np.sqrt(np.mean((sol_train_hf.T - Uh_hf@(Sh_hf.reshape(n_modes, 1)*Vh_hf.T))**2)):.4e}")
    logging.info(f"\nMSE test:  {np.sqrt(np.mean((sol_test_hf.T - Uh_hf@(Sh_hf.reshape(n_modes, 1)*Vh_hf_test.T))**2)):.4e}")
    
    if X_test_coarse1.shape == y_test.shape:
        logging.info(f"\nMSE coarse simulation 1 train:  {np.sqrt(np.mean((sol_train_hf.T - Uh_hf@(Sh_lf.reshape(n_modes, 1)*Vh_lf.T))**2)):.4e}")
        logging.info(f"\nMSE coarse simulation 1 test:  {np.sqrt(np.mean((sol_test_hf.T - Uh_hf@(Sh_hf.reshape(n_modes, 1)*Vh_lf_test.T))**2)):.4e}")

    if X_test_coarse1.shape == y_test.shape:
        logging.info(f"\nMSE coarse simulation 1 train:  {np.sqrt(np.mean((Vh_hf-Vh_lf))**2):.4e}")
        logging.info(f"\nMSE coarse simulation 1 test:  {np.sqrt(np.mean((Vh_hf_test-Vh_lf_test))**2):.4e}")

    # Initialize the multiFidelityNN model
    logging.info("Initializing the MultiFidelityNN model")
    mfnn_model = MultiFidelityNN(
        input_shapes=[(X_train_param.shape[1],),
                      (Vh_lf.shape[1],)],
        merge_mode=config["merge_mode"],
        coeff=config["coeff"],
        layers_config=config["layers_config"],
        train_config=config["train_config"],
        output_units=Vh_hf.shape[1],
        output_activation=config["output_activation"]
    )

    # # Build the model
    logging.info("Building the multiFidelityNN model")
    model=mfnn_model.build_model()
    
    print(model.summary())

    # # Train the model using K-Fold cross-validation
    logging.info("Starting K-Fold training")
    mfnn_model.kfold_train([X_train_param,Vh_lf_scaled], Vh_hf_scaled)
    
    # plot_results(X_test, y_test, mfnn_model.model)
    
    predicted_test = scaler_output.inverse_transform(mfnn_model.model([X_test_param, Vh_lf_test_scaled]))
    reconstructed_test = Uh_hf @ (Sh_hf.reshape(n_modes, 1) * predicted_test.T)
    test_error = np.mean((reconstructed_test.T - sol_test_hf) ** 2)
    print(f"Test Error: {test_error:.6e}")    
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    main()