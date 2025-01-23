import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

def animate_solution(h5_file, sample_idx=0, output_file="reaction_diffusion.mp4"):
    """
    Creates an animation of the solution of the reaction-diffusion system for a given sample.
    
    Args:
        h5_file (str): Path to the HDF5 file containing the dataset.
        sample_idx (int): Index of the sample to animate.
        output_file (str): Path to save the animation video (e.g., 'output.mp4').
    """
    # Load the dataset
    with h5py.File(h5_file, 'r') as h5file:
        # Extract the solution for the specified sample
        d1 = h5file['d1'][sample_idx]
        beta = h5file['beta'][sample_idx]
        t = h5file['t'][:]
        x = h5file['x'][:]
        y = h5file['y'][:]
        u = h5file['u'][sample_idx]  # Shape: (len(t), n, n)
        v = h5file['v'][sample_idx]  # Shape: (len(t), n, n)
    
    # Create a grid for visualization
    X, Y = np.meshgrid(x, y)
    
    # Initialize the figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plt.suptitle(f"Reaction-Diffusion Animation (d1 = {d1:.4f}, beta = {beta:.4f})")

    # Initialize plots for u and v
    u_plot = axs[0].pcolormesh(X, Y, u[0], shading='auto', cmap='jet')
    v_plot = axs[1].pcolormesh(X, Y, v[0], shading='auto', cmap='jet')
    
    axs[0].set_title('u')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[1].set_title('v')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    
    fig.colorbar(u_plot, ax=axs[0])
    fig.colorbar(v_plot, ax=axs[1])

    # Update function for the animation
    def update(frame):
        u_plot.set_array(u[frame].flatten())
        v_plot.set_array(v[frame].flatten())
        axs[0].set_title(f'u at t = {t[frame]:.2f}')
        axs[1].set_title(f'v at t = {t[frame]:.2f}')
        return u_plot, v_plot

    # Create the animation using FFMpegWriter
    writer = FFMpegWriter(fps=10, metadata=dict(artist='ReactionDiffusion'), bitrate=1800)
    with writer.saving(fig, output_file, dpi=100):
        for frame in range(len(t)):
            update(frame)
            writer.grab_frame()
    
    print(f"Animation saved to {output_file}")

# Example usage
if __name__ == "__main__":
    h5_file_path = "reaction_diffusion_training_32x32_coarse.h5"  # Path to the HDF5 file
    sample_index = 0  # Index of the sample to animate
    output_file_path = "reaction_diffusion_sample_0_coarse.mp4"  # Output video file
    
    animate_solution(h5_file_path, sample_idx=sample_index, output_file=output_file_path)