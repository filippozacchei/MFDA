import numpy as np
import scipy.io
from scipy.fftpack import fft2, ifft2
from scipy.integrate import ode
import matplotlib.pyplot as plt

np.random.seed(41) # 41 train, 42 test

# Parameters
num_samples = 500               # Number of LHS samples
d1_range = [0.01, 0.1]
beta_range = [0.5, 1.5]
L = 20                         # Domain size
n = 32                         # Grid size
N = n * n                      # Total grid points
t = np.arange(0, 50.05, 0.5)  # Time vector
output_name = 'reaction_diffusion_training_h4_long.h5'

# Generate LHS samples
lhs_samples = np.random.rand(num_samples, 2)
d1_values = d1_range[0] + lhs_samples[:, 0] * (d1_range[1] - d1_range[0])
beta_values = beta_range[0] + lhs_samples[:, 1] * (beta_range[1] - beta_range[0])

# Spatial grid
x2 = np.linspace(-L / 2, L / 2, n + 1)
x = x2[:-1]
y = x
kx = (2 * np.pi / L) * np.concatenate([np.arange(0, n / 2), np.arange(-n / 2, 0)])
ky = kx
X, Y = np.meshgrid(x, y)
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2
K22 = K2.flatten()

# Dataset storage
dataset = []

def reaction_diffusion_rhs_real(t, uvt, K22, d1, beta, n, N):
    # Separate the real and imaginary parts
    ut_real = uvt[:N].reshape((n, n))
    ut_imag = uvt[N:2*N].reshape((n, n))
    vt_real = uvt[2*N:3*N].reshape((n, n))
    vt_imag = uvt[3*N:].reshape((n, n))
    
    # Reconstruct complex Fourier transforms
    ut = ut_real + 1j * ut_imag
    vt = vt_real + 1j * vt_imag
    
    # Inverse Fourier transform to spatial domain
    u = np.real(ifft2(ut))
    v = np.real(ifft2(vt))
    
    # Reaction terms
    u3 = u**3
    v3 = v**3
    u2v = (u**2) * v
    uv2 = u * (v**2)
    
    # Fourier transform of the RHS
    utrhs = fft2(u - u3 - uv2 + beta * u2v + beta * v3)
    vtrhs = fft2(v - u2v - v3 - beta * u3 - beta * uv2)
    
    # Apply diffusion and reshape for output
    utrhs_real = np.real(utrhs).flatten()
    utrhs_imag = np.imag(utrhs).flatten()
    vtrhs_real = np.real(vtrhs).flatten()
    vtrhs_imag = np.imag(vtrhs).flatten()
    
    rhs_u_real = -d1 * K22 * ut_real.flatten() + utrhs_real
    rhs_u_imag = -d1 * K22 * ut_imag.flatten() + utrhs_imag
    rhs_v_real = -d1 * K22 * vt_real.flatten() + vtrhs_real
    rhs_v_imag = -d1 * K22 * vt_imag.flatten() + vtrhs_imag
    
    return np.concatenate([rhs_u_real, rhs_u_imag, rhs_v_real, rhs_v_imag])

# Main loop for simulations
for i in range(num_samples):
    d1 = d1_values[i]
    beta = beta_values[i]
    print(f"Simulating {i} sample for d1 = {d1:.4f}, beta = {beta:.4f}")
    
    # Initial conditions
    u = np.zeros((n, n, len(t)))
    v = np.zeros((n, n, len(t)))
    u[:, :, 0] = np.tanh(np.sqrt(X**2 + Y**2)) * np.cos(np.angle(X + 1j * Y) - np.sqrt(X**2 + Y**2))
    v[:, :, 0] = np.tanh(np.sqrt(X**2 + Y**2)) * np.sin(np.angle(X + 1j * Y) - np.sqrt(X**2 + Y**2))
    
    # Initial Fourier transform
    ut = fft2(u[:, :, 0])
    vt = fft2(v[:, :, 0])
    uvt_real_imag = np.concatenate([np.real(ut).flatten(), np.imag(ut).flatten(),
                                    np.real(vt).flatten(), np.imag(vt).flatten()])
    
    # Solve the reaction-diffusion system
    solver = ode(reaction_diffusion_rhs_real)
    solver.set_integrator('dopri5')
    solver.set_f_params(K22, d1, beta, n, N)
    solver.set_initial_value(uvt_real_imag, t[0])
    
    uvsol_real_imag = np.zeros((len(t), 4 * N))
    uvsol_real_imag[0] = uvt_real_imag
    for j, t_step in enumerate(t[1:], start=1):
        solver.integrate(t_step)
        uvsol_real_imag[j] = solver.y
    
    # Extract solutions for all time steps
    for j in range(len(t)):
        ut_real = uvsol_real_imag[j, :N].reshape((n, n))
        ut_imag = uvsol_real_imag[j, N:2*N].reshape((n, n))
        vt_real = uvsol_real_imag[j, 2*N:3*N].reshape((n, n))
        vt_imag = uvsol_real_imag[j, 3*N:].reshape((n, n))
        
        ut = ut_real + 1j * ut_imag
        vt = vt_real + 1j * vt_imag
        
        u[:, :, j] = np.real(ifft2(ut))
        v[:, :, j] = np.real(ifft2(vt))
    
    # Store results in dataset
    dataset.append({'d1': d1, 'beta': beta, 't': t, 'x': x, 'y': y, 'u': u, 'v': v})

# # Save dataset to .mat file
# scipy.io.savemat('reaction_diffusion_dataset_full2.mat', {'dataset': dataset})
# print('Dataset saved to reaction_diffusion_dataset_full.mat')


import h5py

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
    t_dset[:] = t
    x_dset[:] = x
    y_dset[:] = y

    # Populate the dataset for each sample
    for i, data in enumerate(dataset):
        d1_dset[i] = data['d1']
        beta_dset[i] = data['beta']
        u_dset[i] = np.transpose(data['u'], (2, 0, 1))  # Transpose (128, 128, 401) -> (401, 128, 128)
        v_dset[i] = np.transpose(data['v'], (2, 0, 1))  # Transpose (128, 128, 401) -> (401, 128, 128)

print("Dataset saved in HDF5 format for training.")

# Visualization function
def visualize_solution(data):
    x = data['x']
    y = data['y']
    u = data['u']
    v = data['v']
    t = data['t']
    
    for j in range(0, len(t), 5):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.pcolormesh(x, y, u[:, :, j], shading='auto', cmap='jet')
        plt.colorbar()
        plt.title(f'u at t = {t[j]:.2f}')
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.subplot(1, 2, 2)
        plt.pcolormesh(x, y, v[:, :, j], shading='auto', cmap='jet')
        plt.colorbar()
        plt.title(f'v at t = {t[j]:.2f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        plt.pause(0.1)
        plt.close()

# Load and visualize some results
visualize_solution(dataset[0])
visualize_solution(dataset[1])