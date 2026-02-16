import matplotlib.pyplot as plt
import arviz as az

def plot_chain_traces(trace, var_names=None, figsize=(12, 4), save_path=None):
    """
    Plot only the MCMC chain time series (no histograms).
    """
    data = az.extract(trace, var_names=var_names)["posterior"]
    n_vars = len(var_names)
    n_chains = data.dims["chain"]

    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)

    if n_vars == 1:
        axes = [axes]

    for i, var in enumerate(var_names):
        for chain in range(n_chains):
            axes[i].plot(data[chain, :, i], alpha=0.7, lw=0.8, label=f"Chain {chain+1}")
        axes[i].set_ylabel(var, fontsize=12)
        axes[i].grid(True, linestyle="--", alpha=0.4)
    axes[-1].set_xlabel("Iteration", fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
    plt.show()
    
file_path = 'samples_1-step_1.nc'  # Replace with your file path
trace = az.from_netcdf(file_path)
plot_chain_traces(trace, var_names=None, figsize=(12, 6), save_path='trace_plot_MFDA.png')