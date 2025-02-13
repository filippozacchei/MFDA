import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
import matplotlib.pyplot as plt

# ----------------------
# 1. Generate Synthetic Data (Parametric Sine Waves)
# ----------------------
num_samples = 1000   # Number of parameter samples
time_steps = 50      # Time series length
param_range = (0.5, 2.0)  # Range of parameter values

# Generate random parameters (frequencies)
params = np.random.uniform(param_range[0], param_range[1], size=(num_samples, 1))

# Generate time values (t) from 0 to 1
t = np.linspace(0, 1, time_steps).reshape(1, time_steps, 1)  # Shape: (1, time_steps, 1)
t = np.tile(t, (num_samples, 1, 1))  # Repeat for all samples (num_samples, time_steps, 1)

# Generate time series using the parametric function y(t) = sin(2π * param * t)
time_series = np.array([np.sin(2 * np.pi * p * t[i, :, 0]) for i, p in enumerate(params)])  # (num_samples, time_steps)

# Reshape inputs for LSTM and FNN
params_repeated = np.tile(params, (1, time_steps)).reshape(num_samples, time_steps, 1)  # (num_samples, time_steps, 1)
X_train = np.concatenate([params_repeated, t], axis=-1)  # (num_samples, time_steps, 2) -> [param, time]
Y_train = time_series.reshape(num_samples, time_steps, 1)  # (num_samples, time_steps, 1)

# ----------------------
# 2. Define LSTM Model
# ----------------------
def build_lstm_model(time_steps, input_dim=2, lstm_units=50):
    input_layer = Input(shape=(time_steps, input_dim), name="LSTM_Input")  # (batch_size, time_steps, 2)

    x = LSTM(lstm_units, activation='tanh', return_sequences=True, name="LSTM_1")(input_layer)
    x = LSTM(lstm_units, activation='tanh', return_sequences=True, name="LSTM_2")(x)
    output = Dense(1, activation='linear', name="TimeSeries_Output")(x)

    model = Model(inputs=input_layer, outputs=output, name="LSTM_Interpolation_Model")
    return model

# ----------------------
# 3. Define Feedforward Neural Network (FNN) Model
# ----------------------
def build_fnn_model(time_steps, input_dim=2, hidden_units=100):
    input_layer = Input(shape=(time_steps, input_dim), name="FNN_Input")  # (batch_size, time_steps, 2)

    x = Dense(hidden_units, activation='relu', name="Dense_1")(input_layer)
    x = Dense(hidden_units, activation='relu', name="Dense_2")(x)
    x = Dense(hidden_units, activation='relu', name="Dense_3")(x)
    output = Dense(1, activation='linear', name="TimeSeries_Output")(x)

    model = Model(inputs=input_layer, outputs=output, name="FNN_Interpolation_Model")
    return model

# ----------------------
# 4. Train LSTM and FNN Models
# ----------------------
lstm_model = build_lstm_model(time_steps)
fnn_model = build_fnn_model(time_steps)

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
fnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\nTraining LSTM model...")
history_lstm = lstm_model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

print("\nTraining FNN model...")
history_fnn = fnn_model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

# ----------------------
# 5. Evaluate and Compare on a New Parameter
# ----------------------
new_param = np.array([[1.2]])  # Example parameter

# Create time values
t_test = np.linspace(0, 1, time_steps).reshape(1, time_steps, 1)

# Repeat the parameter for all time steps
new_param_repeated = np.tile(new_param, (1, time_steps)).reshape(1, time_steps, 1)

# Combine parameter and time inputs
X_test = np.concatenate([new_param_repeated, t_test], axis=-1)  # (1, time_steps, 2)

# Get predictions
predicted_lstm = lstm_model.predict(X_test).flatten()
predicted_fnn = fnn_model.predict(X_test).flatten()
true_series = np.sin(2 * np.pi * new_param[0, 0] * t_test.flatten())  # True function

# ----------------------
# 6. Plot the Results
# ----------------------
plt.figure(figsize=(10, 5))
plt.plot(t_test.flatten(), true_series, label='True', color='black', linewidth=2)
plt.plot(t_test.flatten(), predicted_lstm, label='LSTM Prediction', linestyle='dashed', color='red')
plt.plot(t_test.flatten(), predicted_fnn, label='FNN Prediction', linestyle='dashed', color='blue')
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.legend()
plt.title(f"LSTM vs. FNN for Parametric Time Series Interpolation (param={new_param[0, 0]})")
plt.show()