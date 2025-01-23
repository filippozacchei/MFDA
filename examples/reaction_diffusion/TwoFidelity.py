import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

# Load the boundary dataset
boundary_file = 'reaction_diffusion_boundary_data.h5'
pod_modes_file = 'reaction_diffusion_testing.h5'

with h5py.File(boundary_file, 'r') as h5file:
    u_boundary = h5file['u_boundary'][:]  # Shape: (num_samples, time_steps, boundary_features)
    v_boundary = h5file['v_boundary'][:]  # Shape: (num_samples, time_steps, boundary_features)
    d1_values = h5file['d1'][:]  # Shape: (num_samples,)
    beta_values = h5file['beta'][:]  # Shape: (num_samples,)

with h5py.File(pod_modes_file, 'r') as h5file:
    u_data = h5file['u'][:]  # Shape: (num_samples, time_steps, n, n)
    POD_modes = h5file['POD_modes'][:, :11]  # First 11 modes
    Sigma = h5file['singular_values'][:11]

# Combine boundary data for LSTM input
boundary_data = np.concatenate([u_boundary, v_boundary], axis=-1)  # Shape: (num_samples, time_steps, boundary_features * 2)

# Compute POD coefficients for each time step
num_samples, time_steps, n, _ = u_data.shape
snapshots = u_data.reshape(num_samples * time_steps, n * n).T  # Flatten spatial grid
pod_coefficients = (POD_modes.T @ snapshots) / Sigma[:, None]  # Shape: (11, num_samples * time_steps)
pod_coefficients = pod_coefficients.T.reshape(num_samples, time_steps, 11)  # Reshape to match (samples, time_steps, modes)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(boundary_data, pod_coefficients, test_size=0.2, random_state=42)

# Standardize the inputs and outputs
input_scaler = StandardScaler()
output_scaler = StandardScaler()

# Flatten and scale boundary data
X_train_flat = X_train.reshape(-1, X_train.shape[-1])
X_test_flat = X_test.reshape(-1, X_test.shape[-1])

X_train_scaled = input_scaler.fit_transform(X_train_flat).reshape(X_train.shape)
X_test_scaled = input_scaler.transform(X_test_flat).reshape(X_test.shape)

# Flatten and scale POD coefficients
y_train_flat = y_train.reshape(-1, y_train.shape[-1])
y_test_flat = y_test.reshape(-1, y_test.shape[-1])

y_train_scaled = output_scaler.fit_transform(y_train_flat).reshape(y_train.shape)
y_test_scaled = output_scaler.transform(y_test_flat).reshape(y_test.shape)

# Define the LSTM model
model = Sequential([
    LSTM(64, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, activation='tanh', return_sequences=True),
    Dense(64, activation='relu'),
    Dense(11, activation='linear')  # Predict 11 modes
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Train the model
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_test_scaled, y_test_scaled),
    epochs=100,
    batch_size=32,
    verbose=1
)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
y_pred_scaled = model.predict(X_test_scaled)
y_pred = output_scaler.inverse_transform(y_pred_scaled.reshape(-1, 11)).reshape(y_test.shape)

# Visualize results for a few test samples
for i in range(5):  # Compare 5 samples
    true_modes = y_test[i, -1]  # Last time step's true modes
    pred_modes = y_pred[i, -1]  # Last time step's predicted modes

    print(f"True Modes: {true_modes}")
    print(f"Predicted Modes: {pred_modes}")

    plt.figure(figsize=(10, 5))
    plt.plot(true_modes, label='True Modes', marker='o')
    plt.plot(pred_modes, label='Predicted Modes', marker='x')
    plt.xlabel('Mode Index')
    plt.ylabel('POD Coefficient')
    plt.title(f'Sample {i + 1}: True vs Predicted POD Coefficients')
    plt.legend()
    plt.show()