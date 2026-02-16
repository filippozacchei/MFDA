#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot reference U and V snapshots at selected time steps (HF solution) with sensors.
Publication-ready figure with 2 rows (U, V) and 4 columns (time steps).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys 
# Import your repo utilities
# Add required paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.extend([os.path.join(BASE_DIR, 'forward_models'), os.path.join(BASE_DIR, 'utils')])

from data_utils import load_hdf5
def plot_reference_snapshots(u_test_snapshots, x, y, time_indices, x_data, y_data, n, save_path=None):
    """
    Plot U and V reference fields at selected time indices with sensor grid locations.

    Parameters
    ----------
    u_test_snapshots : np.ndarray
        Array of shape (2*n^2, nt) with stacked U,V reference snapshots.
    x, y : np.ndarray
        1D arrays of spatial grid coordinates.
    time_indices : list of int
        List of time indices to plot (e.g. [0, 500, 1000, 1500]).
    x_data, y_data : np.ndarray
        1D arrays of sensor coordinates in x and y.
    n : int
        Spatial resolution in each direction (grid is n × n).
    save_path : str, optional
        If provided, saves the figure as PDF/PNG.
    """
    nt = u_test_snapshots.shape[1]
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Split U and V from stacked representation
    U_all = u_test_snapshots[:n*n, :].reshape(n, n, nt)
    V_all = u_test_snapshots[n*n:, :].reshape(n, n, nt)

    # Build sensor grid
    Xs, Ys = np.meshgrid(x_data, y_data)

    n_times = len(time_indices)
    fig, axes = plt.subplots(2, n_times, figsize=(4*n_times, 7),
                             constrained_layout=True)

    cmap = "jet"

    for j, t in enumerate(time_indices):
        time_step = t * 0.1
        U = U_all[:, :, t]
        V = V_all[:, :, t]

        # U-field
        ax = axes[0, j]
        im = ax.pcolormesh(X, Y, U, cmap=cmap, shading="auto")
        ax.scatter(
            Xs, Ys,
            facecolor="white", edgecolor="black",
            marker="o", s=35,
            label="Sensors" if j == 0 else ""
        )        
        ax.set_title(f"u, t={time_step}s", fontsize=13)
        ax.set_aspect("equal")

        # V-field
        ax = axes[1, j]
        im = ax.pcolormesh(X, Y, V, cmap=cmap, shading="auto")
        ax.scatter(
            Xs, Ys,
            facecolor="white", edgecolor="black",
            marker="o", s=35,
            label="Sensors" if j == 0 else ""
        )  
        ax.set_title(f"v, t={time_step}s", fontsize=13)
        ax.set_aspect("equal")
    
    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.015, pad=0.02)
    cbar.ax.tick_params(labelsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
            loc="upper left",
            bbox_to_anchor=(0.93, 0.2),   # adjust 0.25 → lower for "below colorbar"
            fontsize=12, frameon=False)
    
    # Formatting
    for ax in axes.ravel():
        ax.tick_params(labelsize=10)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)

    plt.show()


if __name__ == "__main__":
    # Load config file
    config_filepath = "config/config_MultiFidelity_1.json"
    with open(config_filepath, "r") as f:
        config = json.load(f)

    # Load HF training data (for spatial grid)
    train_data = load_hdf5(config["train"])
    x = train_data["x"]
    y = train_data["y"]

    # Load HF test snapshots (u, v)
    u_data = load_hdf5(config["test"])["u"][3]   # shape (1, nt, n, n)
    v_data = load_hdf5(config["test"])["v"][3]

    nt = u_data.shape[0]
    print(nt)
    n = u_data.shape[1]
    print(n)

    # Reshape to stacked (U,V) format: (2*n^2, nt)
    U_flat = u_data.reshape(nt, n*n).T   # (n^2, nt)
    V_flat = v_data.reshape(nt, n*n).T   # (n^2, nt)
    u_test_snapshots = np.vstack([U_flat, V_flat])

    # Define 13 × 13 sensor grid
    x_data = y_data = np.array([-7.5, -6.25, -5.0, -3.75, -2.5, -1.25, 0.0,
                                 1.25,  2.5,  3.75,  5.0,  6.25,  7.5])

    # Choose 4 representative time steps
    time_indices = [0, nt//3+1, 2*nt//3, nt-1]

    # Plot
    plot_reference_snapshots(
        u_test_snapshots=u_test_snapshots,
        x=x,
        y=y,
        time_indices=time_indices,
        x_data=x_data,
        y_data=y_data,
        n=n,
        save_path="reference_fields.pdf"
    )