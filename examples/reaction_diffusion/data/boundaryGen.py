import h5py
import numpy as np

# Input HDF5 dataset file
input_filename = 'reaction_diffusion_training.h5'
output_filename = 'reaction_diffusion_boundary_training.h5'

# Open the input dataset
with h5py.File(input_filename, 'r') as h5file:
    u_data = h5file['u'][:]  # Shape: (num_samples, time_steps, n, n)
    v_data = h5file['v'][:]  # Shape: (num_samples, time_steps, n, n)
    d1_values = h5file['d1'][:]  # Shape: (num_samples,)
    beta_values = h5file['beta'][:]  # Shape: (num_samples,)
    t_values = h5file['t'][:]  # Shape: (time_steps,)

# Extract boundary data
num_samples, time_steps, n, _ = u_data.shape
boundary_features = 4 * 8  # Top, bottom, left, right

# Initialize boundary data arrays
u_boundary = np.zeros((num_samples, time_steps, boundary_features))
v_boundary = np.zeros((num_samples, time_steps, boundary_features))

for i in range(num_samples):
    print(i)
    for j in range(time_steps):
        u_snapshot = u_data[i, j]
        v_snapshot = v_data[i, j]

        # Extract top, bottom, left, and right boundaries
        top_u = u_snapshot[0, ::16]  # Top row
        bottom_u = u_snapshot[-1, ::16]  # Bottom row
        left_u = u_snapshot[::16, 0]  # Left column
        right_u = u_snapshot[::16, -1]  # Right column

        top_v = v_snapshot[0, ::16]
        bottom_v = v_snapshot[-1, ::16]
        left_v = v_snapshot[::16, 0]
        right_v = v_snapshot[::16, -1]

        # Concatenate boundary data for u and v
        u_boundary[i, j] = np.concatenate([top_u, bottom_u, left_u, right_u])
        v_boundary[i, j] = np.concatenate([top_v, bottom_v, left_v, right_v])

# Save the extracted boundary data to a new HDF5 file
with h5py.File(output_filename, 'w') as h5file:
    h5file.create_dataset("u_boundary", data=u_boundary)  # Shape: (num_samples, time_steps, boundary_features)
    h5file.create_dataset("v_boundary", data=v_boundary)  # Shape: (num_samples, time_steps, boundary_features)
    h5file.create_dataset("d1", data=d1_values)  # Shape: (num_samples,)
    h5file.create_dataset("beta", data=beta_values)  # Shape: (num_samples,)
    h5file.create_dataset("t", data=t_values)  # Shape: (time_steps,)

print(f"Boundary data saved to {output_filename}.")