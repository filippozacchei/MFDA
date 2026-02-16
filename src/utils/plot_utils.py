import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def plot_final_U_snapshot(U, x, y, n, title='U', save_path="final_U_plot.png", vmin=None, vmax=None, cmap='jet', n_steps=1000):
    """
    Creates a high-quality static plot of the U quantity at the final time step.

    Parameters:
    - U: np.ndarray, shape (2 * n^2, T), combined U and V fields across time steps.
    - x, y: 1D meshgrid coordinates.
    - n: int, spatial grid resolution (n x n).
    - save_path: str, file path to save the figure.
    - vmin, vmax: float or None, optional color scale limits.
    - cmap: str, colormap for the plot.
    """

    grid_points = n * n
    U_final = U[:grid_points, -1].reshape((n, n))

    fig, ax = plt.subplots(figsize=(5, 4))
    c = ax.pcolormesh(x, y, U_final, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(c, ax=ax, pad=0.02)
    cbar.set_label(title, fontsize=12)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_aspect('equal')

    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Final U snapshot saved to {save_path}")

def plot_2d_pod_modes(U, x, y, n, num_modes=15):
    """Visualizes the first few POD modes."""
    for i in range(num_modes):
        mode = U[:, i].reshape((n, n))
        plt.figure()
        plt.pcolormesh(x, y, mode, shading='auto', cmap='jet')
        plt.colorbar()
        plt.title(f"POD Mode {i+1}")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        

def plot_2d_system_pod_modes(U, x, y, n, num_modes=15):
    """Visualizes the first few POD modes for U and V side by side"""
    grid_points = n * n

    for i in range(num_modes):
        mode_u = U[:grid_points, i].reshape((n, n))  # First half is U
        mode_v = U[grid_points:, i].reshape((n, n))  # Second half is V
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Side-by-side plots

        ax1 = axes[0]
        c1 = ax1.pcolormesh(x, y, mode_u, shading='auto', cmap='jet')
        plt.colorbar(c1, ax=ax1)
        ax1.set_title(f"POD Mode {i+1} (U)")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        ax2 = axes[1]
        c2 = ax2.pcolormesh(x, y, mode_v, shading='auto', cmap='jet')
        plt.colorbar(c2, ax=ax2)
        ax2.set_title(f"POD Mode {i+1} (V)")
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')

        plt.tight_layout()
        plt.show()
        
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_2d_system_prediction(U, x, y, n, n_steps=100, save_path="pod_animation.gif", vmin=-1,vmax=1):
    """Visualizes the first few POD modes for U and V and saves as a GIF (no ffmpeg required)"""
    
    grid_points = n * n

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ax1, ax2 = axes
    c1 = ax1.pcolormesh(x, y, np.zeros((n, n)), shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
    c2 = ax2.pcolormesh(x, y, np.zeros((n, n)), shading='auto', cmap='jet', vmin=vmin, vmax=vmax)

    # Add colorbars
    plt.colorbar(c1, ax=ax1)
    plt.colorbar(c2, ax=ax2)

    ax1.set_title("U")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2.set_title("V")
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # Update function for animation
    def update(i):
        mode_u = U[:grid_points, i].reshape((n, n))  # First half is U
        mode_v = U[grid_points:, i].reshape((n, n))  # Second half is V

        c1.set_array(mode_u.ravel())  # Update U mode
        c2.set_array(mode_v.ravel())  # Update V mode

        ax1.set_title(f"U")
        ax2.set_title(f"V")

        return c1, c2

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=range(0,min(U.shape[1], n_steps),10), interval=100, blit=False
    )

    ani.save(save_path, writer='pillow', fps=5)
    print(f"Animation saved as {save_path}")

    plt.show()

def plot_variance(Sigma):
    """Plots cumulative variance captured by singular values."""
    variance_captured = Sigma**2 / np.sum(Sigma**2)
    cumulative_variance = np.cumsum(variance_captured)
    plt.figure(figsize=(8, 6))
    plt.loglog(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.axhline(0.9, color='r', linestyle='--', label='90% variance')
    plt.axhline(0.95, color='g', linestyle='--', label='95% variance')
    plt.title('Cumulative Variance Captured by POD Modes')
    plt.xlabel('Number of Modes')
    plt.ylabel('Cumulative Variance')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_eigenvalues(Sigma):
    """Plots eigenvalue decay."""
    eigenvalues = Sigma**2
    plt.figure(figsize=(8, 6))
    plt.semilogy(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-')
    plt.title('Eigenvalue Decay')
    plt.xlabel('Mode Number')
    plt.ylabel('Eigenvalue (log scale)')
    plt.grid(True)
    plt.show()
