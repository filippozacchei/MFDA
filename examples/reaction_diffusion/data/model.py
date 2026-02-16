import numpy as np
import scipy.io
from scipy.fftpack import fft2, ifft2
from scipy.integrate import ode
import matplotlib.pyplot as plt
import logging


class Model_DR:
    
    def __init__(self, n=128, dt=0.05, L=20., T=50.05):
        self.L = L                   # Domain size
        self.n = n                   # Grid size
        self.N = n * n               # Total grid points
        self.t = np.arange(0, T, dt) # Time vector
        x2 = np.linspace(-L / 2, L / 2, n + 1)
        self.x = x2[:-1]
        self.y = self.x
        self.kx = (2 * np.pi / L) * np.concatenate([np.arange(0, n / 2), np.arange(-n / 2, 0)])
        self.ky = self.kx
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        self.K2 = self.KX**2 + self.KY**2
        self.K22 = self.K2.flatten()
        self.u = None
        self.v = None

    def reaction_diffusion_rhs_linear_fourier(t, uvt, K22, d1, beta, n, N):
        # Unpack as before
        ut_real = uvt[:N]
        ut_imag = uvt[N:2*N]
        vt_real = uvt[2*N:3*N]
        vt_imag = uvt[3*N:]
        
        # Linear operator in Fourier space: λ(k) = 1 - d1*K^2
        lam = 1.0 - d1 * K22  # shape (N,)
        
        rhs_u_real = lam * ut_real
        rhs_u_imag = lam * ut_imag
        rhs_v_real = lam * vt_real
        rhs_v_imag = lam * vt_imag
        
        return np.concatenate([rhs_u_real, rhs_u_imag, rhs_v_real, rhs_v_imag])

    def reaction_diffusion_rhs_real(self, t, uvt, K22, d1, beta, n, N):
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

    def solve(self, d1, beta):
        self.u = np.zeros((self.n, self.n, len(self.t)))
        self.v = np.zeros((self.n, self.n, len(self.t)))
        self.u[:, :, 0] = np.tanh(np.sqrt(self.X**2 + self.Y**2)) * \
            np.cos(np.angle(self.X + 1j * self.Y) - np.sqrt(self.X**2 + self.Y**2))
        self.v[:, :, 0] = np.tanh(np.sqrt(self.X**2 + self.Y**2)) * \
            np.sin(np.angle(self.X + 1j * self.Y) - np.sqrt(self.X**2 + self.Y**2))
        
        # Initial Fourier transform
        ut = fft2(self.u[:, :, 0])
        vt = fft2(self.v[:, :, 0])
        
        uvt_real_imag = np.concatenate([np.real(ut).flatten(), np.imag(ut).flatten(),
                                        np.real(vt).flatten(), np.imag(vt).flatten()])
        
        # Solve the reaction-diffusion system
        solver = ode(self.reaction_diffusion_rhs_real)
        solver.set_f_params(self.K22, d1, beta, self.n, self.N)
        solver.set_initial_value(uvt_real_imag, self.t[0])
        
        uvsol_real_imag = np.zeros((len(self.t), 4 * self.N))
        uvsol_real_imag[0] = uvt_real_imag
        for j, t_step in enumerate(self.t[1:], start=1):
            solver.integrate(t_step)
            uvsol_real_imag[j] = solver.y
        
        # Extract solutions for all time steps
        for j in range(len(self.t)):
            print
            ut_real = uvsol_real_imag[j, :self.N].reshape((self.n, self.n))
            ut_imag = uvsol_real_imag[j, self.N:2*self.N].reshape((self.n, self.n))
            vt_real = uvsol_real_imag[j, 2*self.N:3*self.N].reshape((self.n, self.n))
            vt_imag = uvsol_real_imag[j, 3*self.N:].reshape((self.n, self.n))
            
            ut = ut_real + 1j * ut_imag
            vt = vt_real + 1j * vt_imag
            
            self.u[:, :, j] = np.real(ifft2(ut))
            self.v[:, :, j] = np.real(ifft2(vt))

        return self.u,self.v
    
    def get_datapoint(self, x, y):
        """
        Extracts the time evolution of variables u and v at the closest grid point to (x, y).
        
        :param x: X-coordinate (float)
        :param y: Y-coordinate (float)
        :return: Time evolution of u and v at the specified coordinates
        """
        # Find the nearest indices in the grid
        i = np.argmin(np.abs(self.x - x))
        j = np.argmin(np.abs(self.y - y))
        
        # Return the time evolution at the nearest grid point
        return self.u[i, j, :], self.v[i, j, :]
    
    def get_data(self, coordinates, lag=1):
        """
        Extracts the time evolution of variables u and v at the given coordinates.

        :param coordinates: Array of (x, y) coordinate pairs.
        :return: Array of shape (num_points, 2, num_time_steps) containing u and v data.
        """
        # Convert coordinates to numpy array
        coordinates = np.array(coordinates)

        # Find the nearest indices in the grid
        i_indices = np.array([np.argmin(np.abs(self.x - x)) for x in coordinates[:, 0]])
        j_indices = np.array([np.argmin(np.abs(self.y - y)) for y in coordinates[:, 1]])

        # Preallocate result array
        num_points = len(coordinates)
        num_time_steps = self.u.shape[2]
        data = np.empty((num_points, 2,1+(num_time_steps-1)//lag), dtype=np.float64)

        # Efficiently gather the data for all points
        for idx in range(num_points):
            data[idx, 0, :] = self.u[i_indices[idx], j_indices[idx], ::lag]
            data[idx, 1, :] = self.v[i_indices[idx], j_indices[idx], ::lag]

        return data
