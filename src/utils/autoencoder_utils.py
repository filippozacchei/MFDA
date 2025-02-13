import h5py
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler

def reshape_to_autoencoder_input(data):
    """Reshapes the dataset into a 2D input matrix for autoencoder training."""
    num_samples, time_steps, n, _ = data.shape
    grid_points = n * n
    return data.reshape(num_samples * time_steps, grid_points)

def reshape_to_conv_autoencoder_input(data):
    """Reshapes the dataset into a 2D input matrix for autoencoder training."""
    num_samples, time_steps, n, _ = data.shape
    grid_points = n * n
    return data.reshape(num_samples * time_steps, n,n)

def reshape_to_lstm(data, num_samples: int, time_steps: int, num_modes: int):
    """Reshapes the dataset into a 3D matrix for LSTM input."""
    return data.reshape(num_samples, time_steps, num_modes)

def dense_block(x, units, dropout_rate=0.0,activation='relu'):
    """Creates a dense layer block with batch normalization and optional dropout."""
    x = Dense(units, activation=activation)(x)
    if dropout_rate:
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    return x

def build_autoencoder(input_dim, latent_dim):
    """
    Builds an optimized autoencoder with a cleaner architecture.
    
    :param input_dim: Dimensionality of input data.
    :param latent_dim: Dimensionality of the encoded latent space.
    :return: Tuple (autoencoder, encoder, decoder).
    """
    logging.info("Building optimized autoencoder model.")
    
    # Encoder
    input_layer = Input(shape=(input_dim,))
    x = dense_block(input_layer, 2048, activation='relu')
    x = dense_block(x, 2048, activation='relu')
    encoded = dense_block(x, latent_dim, activation='linear')

    # Decoder
    x = dense_block(encoded, 2048, dropout_rate=0.0, activation='relu')
    x = dense_block(x, 2048, dropout_rate=0.0, activation='relu')
    decoded = Dense(input_dim, activation='linear')(x)

    # Define models
    autoencoder = Model(input_layer, decoded, name='Autoencoder')
    encoder = Model(input_layer, encoded, name='Encoder')
    decoder = Model(encoded, decoded, name='Decoder')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Use fixed learning rate
    autoencoder.compile(optimizer=optimizer, loss='mse')
    logging.info("Optimized Autoencoder model compiled successfully with learning rate decay.")

    return autoencoder, encoder, decoder

def lr_decay(epoch, lr):
    """Applies an exponential decay to the learning rate: lr * 0.99"""
    return lr * 0.99

def train_autoencoder(autoencoder, x_train, x_val=None, epochs=200, batch_size=64):
    """
    Trains the autoencoder with an improved strategy using early stopping and learning rate scheduling.
    
    :param autoencoder: Compiled autoencoder model.
    :param x_train: Training data.
    :param x_val: Validation data (optional).
    :param epochs: Number of training epochs.
    :param batch_size: Batch size for training.
    :return: Trained autoencoder model.
    """
    logging.info("Training autoencoder with early stopping and learning rate reduction.")
    
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_decay),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    history = autoencoder.fit(
        x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, x_val) if x_val is not None else None,
        callbacks=callbacks,
        verbose=1
    )
    
    logging.info("Autoencoder training completed.")
    return history


def reconstruct_autoencoder(encoder, decoder, data):
    """Encodes and decodes data using the trained autoencoder."""
    latent_representation = encoder.predict(data)
    reconstructed = decoder.predict(latent_representation)
    return reconstructed

def save_autoencoder_models(encoder, decoder, resolution):
    """Saves encoder and decoder models based on resolution."""
    encoder_path = f"encoder_{resolution}.h5"
    decoder_path = f"decoder_{resolution}.h5"
    
    logging.info(f"Saving encoder model to {encoder_path}")
    encoder.save(encoder_path)
    
    logging.info(f"Saving decoder model to {decoder_path}")
    decoder.save(decoder_path)
    
    logging.info("Autoencoder models saved successfully.")

def load_autoencoder_models(resolution):
    """Loads encoder and decoder models based on resolution."""
    encoder_path = f"encoder_{resolution}.h5"
    decoder_path = f"decoder_{resolution}.h5"
    
    logging.info(f"Loading encoder model from {encoder_path}")
    encoder = tf.keras.models.load_model(encoder_path)
    
    logging.info(f"Loading decoder model from {decoder_path}")
    decoder = tf.keras.models.load_model(decoder_path)
    
    return encoder, decoder

