import pandas as pd
import numpy as np
import tensorflow as tf
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import tensorflow as tf

modes = 5

def make_scheduler(coeff: float, mode: str = 'linear', step=10):
    """Creates a learning rate scheduler."""
    if mode == 'linear':
        def scheduler(epoch: int, lr: float) -> float:
            return lr if epoch < step else max(lr * coeff, 1e-7)
        return scheduler
    elif mode == 'decay':
        def scheduler(epoch: int, lr: float) -> float:
            return lr * (1 + coeff * epoch) / (1 + coeff * (epoch + 1))
        return scheduler
    else:
        raise ValueError(f"Unsupported mode: {mode}. Only 'linear' and 'decay' are supported.")

import h5py
import numpy as np

# Load the HDF5 dataset
filename = './data/reaction_diffusion_training.h5'
with h5py.File(filename, 'r') as h5file:
    u_data = h5file['u'][:]  # Shape: (num_samples, len(t), n, n)

print(f"Loaded dataset with {u_data.shape[0]} samples.")

# Load the HDF5 dataset
filename = './data/reaction_diffusion_testing.h5'

with h5py.File(filename, 'r') as h5file:
    u_data_test      = h5file['u'][:]  # Shape: (num_samples, len(t), n, n)

print(f"Loaded dataset with {u_data_test.shape[0]} samples.")


# Load POD basis (modes and singular values)
with h5py.File('./data/pod_basis.h5', 'r') as h5file:
    POD_modes = h5file['POD_modes'][:, :66]  # Use first 34 modes
    Sigma = h5file['singular_values'][:66]  # Singular values

# Load input features for training and testing datasets
training_data = pd.read_csv('./data/pod_coefficients_training_dataset.csv')  # Only inputs
testing_data = pd.read_csv('./data/pod_coefficients_testing_dataset.csv')  # Only inputs

# Reshape `u_data` and `u_data_test` to compute POD coefficients
num_samples_train, time_steps_train, n, _ = u_data.shape
snapshots_train = u_data.reshape(num_samples_train * time_steps_train, n * n).T

num_samples_test, time_steps_test, n, _ = u_data_test.shape
snapshots_test = u_data_test.reshape(num_samples_test * time_steps_test, n * n).T

# Compute POD coefficients (targets) by projecting onto POD modes
train_coefficients = (POD_modes.T @ snapshots_train) / Sigma[:, None]
test_coefficients = (POD_modes.T @ snapshots_test) / Sigma[:, None]

# Combine inputs with computed targets
X_train = training_data[['d1', 'beta', 'time']].values
y_train = train_coefficients.T  # POD coefficients as targets

X_test = testing_data[['d1', 'beta', 'time']].values
y_test = test_coefficients.T  # POD coefficients as targets

# Standardize inputs and outputs
input_scaler = StandardScaler()
output_scaler = StandardScaler()

X_train = input_scaler.fit_transform(X_train)
X_test = input_scaler.transform(X_test)

y_train_scaled = output_scaler.fit_transform(y_train[:, :modes])  # First 5 modes
y_test_scaled = output_scaler.transform(y_test[:, :modes])

print(y_train.shape)

# # Define the neural network
# model = Sequential([
#     Dense(128, activation='gelu', input_shape=(X_train.shape[1],)),
#     Dense(128, activation='gelu'),
#     Dense(128, activation='gelu'),
#     Dense(128, activation='gelu'),
#     Dense(128, activation='gelu'),
#     Dense(128, activation='gelu'),
#     Dense(128, activation='gelu'),
#     Dense(128, activation='gelu'),
#     Dense(y_train[:,:modes].shape[1], activation='linear')  # Output layer with 'num_modes' neurons
# ])
# # Compile the model
# model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
# model.summary()

# # Train the model
# history = model.fit(
#     X_train, y_train_scaled,
#     validation_data=(X_test, y_test_scaled),
#     epochs=50,
#     batch_size=1000,
#     callbacks=[LearningRateScheduler(make_scheduler(0.99))],
#     verbose=1
# )

# model.save("pod_nn_model.keras")

# # Plot training and validation loss
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

model = tf.keras.models.load_model("pod_nn_model.keras")

# Reconstruct and compare solutions for the test dataset
for i in range(500):  # Compare first 5 samples
    X_sample = X_test[i*100]  # Input sample
    y_sample_true = y_test[i*100]  # True POD coefficients
    y_sample_pred = model.predict(X_sample.reshape(1, -1)).flatten()  # Predicted coefficients
    u_true = snapshots_test[:,i*100]

    # Reconstruct solutions
    u_reconstructed_pred = POD_modes[:, :modes] @ (Sigma[:modes] * output_scaler.inverse_transform(y_sample_pred.reshape(1, -1))).T
    u_reconstructed_true = POD_modes @ (Sigma * y_sample_true).T
    u_reconstructed_5 = POD_modes[:, :modes] @ (Sigma[:modes] * y_sample_true[:modes]).T

    # Reshape for plotting
    n = int(np.sqrt(POD_modes.shape[0]))
    u_reconstructed_pred = u_reconstructed_pred.reshape((n, n))
    u_reconstructed_true = u_reconstructed_true.reshape((n, n))
    u_reconstructed_5 = u_reconstructed_5.reshape((n, n))
    u_true = u_true.reshape((n, n))

    # Plot comparison
    plt.figure(figsize=(12, 12))  # Adjusted figure size for better visibility

    # Predicted solution
    plt.subplot(2, 2, 1)
    plt.pcolormesh(u_reconstructed_pred, shading='auto', cmap='jet')
    plt.colorbar()
    plt.title('Predicted Solution')

    # Projected solution
    plt.subplot(2, 2, 2)
    plt.pcolormesh(u_reconstructed_true, shading='auto', cmap='jet')
    plt.colorbar()
    plt.title('Projected Solution')

    # 5 Modes solution
    plt.subplot(2, 2, 3)
    plt.pcolormesh(u_reconstructed_5, shading='auto', cmap='jet')
    plt.colorbar()
    plt.title('5 Modes Solution')

    # True solution
    plt.subplot(2, 2, 4)
    plt.pcolormesh(u_true, shading='auto', cmap='jet')
    plt.colorbar()
    plt.title('True Solution')

    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()