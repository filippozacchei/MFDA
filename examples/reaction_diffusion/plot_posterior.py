import matplotlib.pyplot as plt
import arviz as az

def plot_posterior_histograms(files, labels, var_names, true_values=None, bins=30, figsize=(12, 5), save_path=None):
    """
    Plot posterior histograms for multiple cases side by side.

    Parameters:
    - files: list of paths to .nc files (ArviZ InferenceData)
    - labels: list of case labels (e.g., ["1-step", "MLDA", "FOM"])
    - var_names: list of parameter names to plot
    - true_values: list/array of true parameter values (same length as var_names)
    - bins: number of histogram bins
    - figsize: figure size
    - save_path: optional path to save the figure
    """
    n_cases = len(files)
    n_vars = len(var_names)
    fig, axes = plt.subplots(n_vars, n_cases, figsize=figsize, sharey="row", constrained_layout=True)

    if n_vars == 1:  # ensure axes are always 2D
        axes = axes[None, :]

    for j, (file, label) in enumerate(zip(files, labels)):
        trace = az.from_netcdf(file)
        for i, var in enumerate(var_names):
            # extract flattened samples
            samples = trace.posterior[var].values.flatten()
            
            ax = axes[i, j]
            ax.hist(samples, bins=bins, color="#56B4E9", edgecolor="k", alpha=0.7, density=True)
            
            if true_values is not None:
                ax.axvline(true_values[i], color="red", linestyle="--", linewidth=2, label="True value" if j==0 else None)

            if i == 0:
                ax.set_title(label, fontsize=14)
            if j == 0:
                ax.set_ylabel(var, fontsize=12)
            ax.tick_params(labelsize=10)

    # Add legend once
    if true_values is not None:
        axes[0, -1].legend(loc="upper right", fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
    plt.show()


# Example usage
files = [
    "samples_1-step_1.nc",
    "samples_MLDA1.nc",
    "samples_FOM.nc"
]
labels = ["MFDA", "MLDA", "FOM"]
var_names = [r"$\mu_0$", r"$\mu_1$"]   # replace with your actual parameter names
true_params = [0.07268223, 0.78613933]

plot_posterior_histograms(files, labels, var_names, true_values=true_params, figsize=(12, 5), save_path="posterior_histograms_MFDA.png")