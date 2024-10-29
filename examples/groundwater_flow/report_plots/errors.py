import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.models import load_model
import sys
sys.path.append('../')
import os
from utils import prepare_data_single_fidelity

# Parameters
n_splits = 4
n_samples = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
discretizations = ['h1', 'h2', 'h3']
resolutions = [50, 25, 10]
colors = {'h1': 'blue', 'h2': 'green', 'h3': 'red'}
markers = {'h1': 'o', 'h2': 's', 'h3': 'D'}
linestyles = {'h1': '-', 'h2': '--', 'h3': ':'}

# Prepare dictionary for RMSE results
rmse_test_all = {disc: [] for disc in discretizations}

# Iterate over discretizations
for discretization in discretizations:
    print(f"Processing discretization level: {discretization}")

    # Load training and test data for each discretization
    X_train, y_train, X_test, y_test = prepare_data_single_fidelity(
        115200,
        f"../data/X_train_{discretization}.csv",
        f"../data/y_train_{discretization}.csv",
        f"../data/X_test_{discretization}.csv",
        f"../data/y_test_{discretization}.csv"
    )

    for n_sample in n_samples:
        kf = KFold(n_splits=n_splits)
        X, y = X_train[:n_sample, :], y_train[:n_sample, :]
        rmse_test = []
        predictions_train = []
        predictions_test = []

        # Perform k-fold cross-validation
        for fold_var, (train_index, val_index) in enumerate(kf.split(X), start=1):

            # Load model and evaluate RMSE
            model_path = f'../models/single_fidelity/resolution_{discretization}/samples_{n_sample}/model_fold_{fold_var}.keras'
            model = load_model(model_path)
            predictions_train.append(model(X_train).numpy())
            predictions_test.append(model(X_test).numpy())

            rmse_test.append(np.sqrt(np.mean((model(X_test).numpy() - y_test) ** 2)))
        

        predictions_train = np.mean(predictions_train, axis=0)
        predictions_test = np.mean(predictions_test, axis=0)

        # Define the directory path
        output_dir = f'../data/predictions_single_fidelity/resolution_{discretization}/samples_{n_sample}/'

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the files
        np.savetxt(os.path.join(output_dir, 'predictions_train.csv'), predictions_train, delimiter=",")
        np.savetxt(os.path.join(output_dir, 'predictions_test.csv'), predictions_test, delimiter=",")

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

# Plot theoretical convergence line
sample_positions = np.arange(1, len(n_samples) * (len(discretizations) + 1), len(discretizations) + 1) + 1
convergence_line = 1.10 * np.sqrt(64000) / np.sqrt(n_samples) * np.mean(rmse_test_all['h2'][-1])
plt.plot(sample_positions, convergence_line, 'k--')

# Final adjustments to plot
plt.xticks(np.arange(1, len(n_samples) * (len(discretizations) + 1), len(discretizations) + 1) + 1, n_samples)
plt.xlabel('Number of Data Samples', fontsize=14)
plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=14)
plt.legend(custom_lines + [plt.Line2D([0], [0], color='black', linestyle='--', lw=2)],
           [f'{r}-point Discretization' for r in resolutions] + [r'$\propto \frac{1}{\sqrt{n}}$'], 
           loc='upper right', fontsize=12, frameon=False)

plt.grid(True, linestyle='--', alpha=0.7)
plt.yscale('log')
plt.tight_layout()
plt.savefig("../images/convergence_error_single_fidelity.png", dpi=300, bbox_inches='tight')
