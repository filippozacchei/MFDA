import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.models import load_model
import sys
sys.path.append('../')
import os
from utils import *

# Parameters
n_splits = 4
n_samples = [1000, 2000, 4000, 8000, 16000, 32000]
discretizations = ['','_no_level0']
resolutions = [50, 25, 10]
colors = {'': 'blue', '_no_level0': 'green'}
markers = {'': 'o', '_no_level0': 's'}
linestyles = {'': '-', '_no_level0': '--'}

# Prepare dictionary for RMSE results
rmse_test_all = {disc: [] for disc in discretizations}

# Iterate over discretizations
for discretization in discretizations:
    print(f"Processing discretization level: {discretization}")

    for n_sample in n_samples:

        X_train_param, X_train_coarse, X_train_nn_level0, y_train = prepare_data_multi_fidelity(
            n_sample, 
            f"../data/y_train_h1.csv", 
            f"../data/X_train_h1.csv", 
            f"../data/y_train_h2.csv", 
            f"../data/predictions_single_fidelity/resolution_h1/samples_{n_sample}/predictions_train.csv",
            )
        X_test_param, X_test_coarse, X_test_nn_level0, y_test = prepare_data_multi_fidelity(
            12800,
            f"../data/y_test_h1.csv", 
            f"../data/X_test_h1.csv", 
            f"../data/y_test_h2.csv", 
            f"../data/predictions_single_fidelity/resolution_h1/samples_{n_sample}/predictions_test.csv",
        )

        kf = KFold(n_splits=n_splits)
        X, y = X_train_param[:n_sample, :], y_train[:n_sample, :]
        rmse_test = []

        # Perform k-fold cross-validation
        for fold_var, (train_index, val_index) in enumerate(kf.split(X), start=1):

            # Load model and evaluate RMSE
            model_path = f'../models/multi_fidelity/resolution_50-25{discretization}/samples_{n_sample}/model_fold_{fold_var}.keras'
            model = load_model(model_path)

            if discretization == '':
                rmse_test.append(np.sqrt(np.mean((model((X_test_param,X_test_coarse,X_test_nn_level0)).numpy() - y_test) ** 2)))
            else:
                rmse_test.append(np.sqrt(np.mean((model((X_test_param,X_test_coarse)).numpy() - y_test) ** 2)))

        
        # predictions_train = model(X_train).numpy()
        # predictions_test = model(X_test).numpy()

        # # Define the directory path
        # output_dir = f'../data/predictions_single_fidelity/resolution_{discretization}/samples_{n_sample}/'

        # # Create the directory if it doesn't exist
        # os.makedirs(output_dir, exist_ok=True)

        # # Save the files
        # np.savetxt(os.path.join(output_dir, 'predictions_train.csv'), predictions_train, delimiter=",")
        # np.savetxt(os.path.join(output_dir, 'predictions_test.csv'), predictions_test, delimiter=",")

        # Append RMSE results for current sample size and discretization
        rmse_test_all[discretization].append(rmse_test)

# Plotting
plt.figure(figsize=(12, 8))
custom_lines = [plt.Line2D([0], [0], color=colors[d], lw=4, linestyle=linestyles[d], marker=markers[d]) for d in discretizations]

# Create boxplots for each discretization level and sample size
for i, discretization in enumerate(discretizations):
    positions = np.arange(len(n_samples)) * (len(discretizations) + 1) + i + 1
    plt.boxplot(rmse_test_all[discretization], positions=positions, widths=0.6, patch_artist=True,
                boxprops=dict(facecolor=colors[discretization], edgecolor='none'),
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(color=colors[discretization], linewidth=1.5),
                capprops=dict(color=colors[discretization], linewidth=1.5),
                flierprops=dict(marker=markers[discretization], color=colors[discretization], alpha=0.7, markersize=8),
                showfliers=True)

# Plot convergence lines for each discretization level
for i, d in enumerate(discretizations):
    pos = np.arange(len(n_samples)) * (len(discretizations) + 1) + i + 1
    plt.plot(pos, np.mean(rmse_test_all[d], axis=1), color=colors[d], linestyle=linestyles[d])


# Final adjustments to plot
plt.xticks(np.arange(1, len(n_samples) * (len(discretizations) + 1), len(discretizations) + 1) + 1, n_samples)
plt.xlabel('Number of Data Samples', fontsize=14)
plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=14)
plt.legend(custom_lines,
           [f'{r}-point Discretization' for r in resolutions], 
           loc='upper right', fontsize=12, frameon=False)

plt.grid(True, linestyle='--', alpha=0.7)
plt.yscale('log')
plt.tight_layout()
plt.savefig("../images/convergence_error_multi_fidelity.png", dpi=300, bbox_inches='tight')
