import sys
import os
import json
import logging
import tensorflow as tf
import numpy as np
# Add the 'models' directory to the Python path
# Add the 'models' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/forward_models/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/utils/'))

from multi_fidelity_nn import MultiFidelityNN
from data_utils import *

# Set up logging for progress output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data(config):
    """
    Prepare the training and testing datasets.
    :param config: Configuration dictionary.
    :return: X_train, y_train, X_test, y_test arrays.
    """
    logging.info("Preparing data for multi-fidelity model")
    X_train_param, X_train_coarse1, X_train_coarse2, y_train = prepare_data_multi_fidelity(
        config['n_sample'], 
        config["y_train"],
        config["X_train_param"], 
        config["X_train_coarse1"],
        config["X_train_coarse2"]
        )
    X_test_param, X_test_coarse1, X_test_coarse2, y_test = prepare_data_multi_fidelity(
        12800,
        config["y_test"],
        config["X_test_param"], 
        config["X_test_coarse1"],
        config["X_test_coarse2"]
    )
    return X_train_param, X_train_coarse1, X_train_coarse2, y_train, X_test_param, X_test_coarse1, X_test_coarse2, y_test

def main():         
    # Add the 'models' directory to the Python path
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/forward_models/'))

    # Load configuration
    config_filepath = 'config/config_MultiFidelity2.json'
    config = load_config(config_filepath)

    destination_folder = config["train_config"]["model_save_path"]
    shutil.copy(config_filepath, destination_folder)

    # Prepare training and testing data
    X_train_param, X_train_coarse1, X_train_coarse2, y_train, X_test_param, X_test_coarse1, X_test_coarse2, y_test = prepare_data(config)
    
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
    if X_test_coarse1.shape == y_test.shape:
        logging.info(f"\nMSE coarse simulation 1 train:  {np.sqrt(np.mean((X_train_coarse1 - y_train)**2)):.4e}")
        logging.info(f"\nMSE coarse simulation 1 test:  {np.sqrt(np.mean((X_test_coarse1 - y_test)**2)):.4e}")

    if X_test_coarse2.shape == y_test.shape:
        logging.info(f"\nMSE coarse simulation 2 train:  {np.sqrt(np.mean((X_train_coarse2 - y_train)**2)):.4e}")
        logging.info(f"\nMSE coarse simulation 2 test:  {np.sqrt(np.mean((X_test_coarse2 - y_test)**2)):.4e}")

    # Initialize the multiFidelityNN model
    logging.info("Initializing the MultiFidelityNN model")
    mfnn_model = MultiFidelityNN(
        input_shapes=[(X_train_param.shape[1],),
                      (X_train_coarse1.shape[1],),
                      (X_train_coarse2.shape[1],)],
        merge_mode=config["merge_mode"],
        coeff=config["coeff"],
        layers_config=config["layers_config"],
        train_config=config["train_config"],
        output_units=y_train.shape[1],
        output_activation=config["output_activation"]
    )

    # # Build the model
    logging.info("Building the multiFidelityNN model")
    model=mfnn_model.build_model()
    
    print(model.summary())

    # # Train the model using K-Fold cross-validation
    logging.info("Starting K-Fold training")
    #mfnn_model.train([X_train_param,X_train_coarse1,X_train_coarse2], y_train, [X_test_param,X_test_coarse1, X_test_coarse2],y_test)
    
    # plot_results(X_test, y_test, mfnn_model.model)
    
    print("Test Error: ", np.mean((mfnn_model.model([X_test_param,X_test_coarse1,X_test_coarse2])-y_test)**2))
    
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    main()
