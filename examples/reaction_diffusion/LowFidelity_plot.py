import os
import sys
import json
import logging
import shutil
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Local Module Imports
sys.path.append('./data/')
sys.path.append('./models/')

# Add required paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.extend([os.path.join(BASE_DIR, 'forward_models'), os.path.join(BASE_DIR, 'utils')])
from model import Model_DR
# Import necessary modules
from multi_fidelity_lstm import MultiFidelityLSTM
from pod_utils import reshape_to_pod_2d_system_snapshots, project, reshape_to_lstm, reconstruct
from data_utils import load_hdf5, prepare_lstm_dataset
from plot_utils import plot_final_U_snapshot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
def load_configuration(config_path):
    """Load JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)
    
def temporal_interpolation_splines(u_data_coarse, time_steps_coarse, time_steps_fine):
    """ Perform temporal interpolation on u_data_coarse using cubic splines
    to match the time dimensionality of the high-fidelity data. """
    num_samples, _, n, n = u_data_coarse.shape
    # Create a normalized time grid for coarse and fine data
    time_coarse = np.linspace(0, 1, time_steps_coarse)
    time_fine = np.linspace(0, 1, time_steps_fine)
    
    # Allocate memory for interpolated data
    u_data_coarse_interpolated = np.zeros((num_samples, time_steps_fine, n, n))
    
    # Perform interpolation for each sample and spatial location
    for sample_idx in range(num_samples):
        for i in range(n):
            for j in range(n):
                # Extract the time series for the current spatial location
                time_series = u_data_coarse[sample_idx, :, i, j]
                # Create a cubic spline interpolator
                spline = interp1d(time_coarse, time_series, kind='cubic', fill_value="extrapolate")
                # Interpolate to the fine time grid
                u_data_coarse_interpolated[sample_idx, :, i, j] = spline(time_fine)
    
    return u_data_coarse_interpolated


def load_and_process_data(config, num_modes=40):
    """
    Load datasets, apply POD projection, and prepare LSTM-ready data.
    :param config: Configuration dictionary.
    :param num_modes: Number of POD modes to retain.
    :return: Processed training and testing data.
    """

    logging.info("Loading datasets...")
    test_data = load_hdf5(config["test"])
    pod_basis = load_hdf5(config["pod_basis"])
    print(test_data['beta'])
    print(test_data['d1'])
    test_data_coarse1 = load_hdf5(config["test_coarse1"])
    pod_basis_coarse1 = load_hdf5(config["pod_basis_coarse1"])

    test_data_coarse2 = load_hdf5(config["test_coarse2"])
    pod_basis_coarse2 = load_hdf5(config["pod_basis_coarse2"])

    test_data_coarse3 = load_hdf5(config["test_coarse3"])
    pod_basis_coarse3 = load_hdf5(config["pod_basis_coarse3"])

    test_data_coarse1['u'] = temporal_interpolation_splines(
        test_data_coarse1['u'], test_data_coarse1['u'].shape[1], test_data['u'].shape[1]
    )

    test_data_coarse1['v'] = temporal_interpolation_splines(
        test_data_coarse1['v'], test_data_coarse1['v'].shape[1], test_data['v'].shape[1]
    )

    test_data_coarse2['u'] = temporal_interpolation_splines(
        test_data_coarse2['u'], test_data_coarse2['u'].shape[1], test_data['u'].shape[1]
    )
    
    test_data_coarse2['v'] = temporal_interpolation_splines(
        test_data_coarse2['v'], test_data_coarse2['v'].shape[1], test_data['v'].shape[1]
    )

    test_data_coarse3['u'] = temporal_interpolation_splines(
        test_data_coarse3['u'], test_data_coarse3['u'].shape[1], test_data['u'].shape[1]
    )

    test_data_coarse3['v'] = temporal_interpolation_splines(
        test_data_coarse3['v'], test_data_coarse3['v'].shape[1], test_data['v'].shape[1]
    )


    # Reshape data into 2D POD snapshots
    logging.info("Reshaping datasets into 2D POD snapshots.")
    u_test_snapshots = reshape_to_pod_2d_system_snapshots(test_data['u'], test_data['v'])
    u_test_snapshots_coarse1 = reshape_to_pod_2d_system_snapshots(test_data_coarse1['u'], test_data_coarse1['v'])
    u_test_snapshots_coarse2 = reshape_to_pod_2d_system_snapshots(test_data_coarse2['u'], test_data_coarse2['v'])
    u_test_snapshots_coarse3 = reshape_to_pod_2d_system_snapshots(test_data_coarse3['u'], test_data_coarse3['v'])

    return u_test_snapshots, u_test_snapshots_coarse1, u_test_snapshots_coarse2, u_test_snapshots_coarse3, test_data, test_data_coarse1, test_data_coarse2, test_data_coarse3

def main():
    """Main execution function."""
    config_filepath = 'config/config_MultiFidelity_3.json'
    config = load_configuration(config_filepath)

    u_test_snapshots, u_test_snapshots_coarse1, u_test_snapshots_coarse2, u_test_snapshots_coarse3, test_data, test_data_coarse1, test_data_coarse2, test_data_coarse3 = load_and_process_data(config, num_modes=40)
    nt=2000
    plot_final_U_snapshot(u_test_snapshots[:,:nt], test_data['x'], test_data['y'], test_data['x'].shape[0], n_steps=1001, save_path='./gif/exact_hf.pdf')
    plot_final_U_snapshot(u_test_snapshots_coarse1[:,:nt], test_data_coarse1['x'], test_data_coarse1['y'], test_data_coarse1['x'].shape[0], n_steps=1001, save_path='./gif/exact_lf1.pdf')
    plot_final_U_snapshot(u_test_snapshots_coarse2[:,:nt], test_data_coarse2['x'], test_data_coarse2['y'], test_data_coarse2['x'].shape[0], n_steps=1001, save_path='./gif/exact_lf2.pdf')
    plot_final_U_snapshot(u_test_snapshots_coarse3[:,:nt], test_data_coarse3['x'], test_data_coarse3['y'], test_data_coarse3['x'].shape[0], n_steps=1001, save_path='./gif/exact_lf3.pdf')

    solver_h1 = Model_DR(n=128, dt=0.05, L=20., T=50.05)
    i_indices_lf1 = np.array([np.argmin(np.abs(solver_h1.x - x)) for x in test_data_coarse1['x']])
    j_indices_lf1 = np.array([np.argmin(np.abs(solver_h1.y - y)) for y in test_data_coarse1['y']])
    i_indices_lf2 = np.array([np.argmin(np.abs(solver_h1.x - x)) for x in test_data_coarse2['x']])
    j_indices_lf2 = np.array([np.argmin(np.abs(solver_h1.y - y)) for y in test_data_coarse2['y']])
    i_indices_lf3 = np.array([np.argmin(np.abs(solver_h1.x - x)) for x in test_data_coarse3['x']])
    j_indices_lf3 = np.array([np.argmin(np.abs(solver_h1.y - y)) for y in test_data_coarse3['y']])

    def get_data(sol_u, sol_v, i_indices, j_indices, lag=1):
        sol_u_sub = sol_u[:, :, :, ::lag]
        sol_v_sub = sol_v[:, :, :, ::lag]
        return  sol_u_sub[:,:,i_indices][:,:,:,j_indices], sol_v_sub[:,:,i_indices][:,:,:,j_indices]   # Same shape

    hf_coarse1_u, hf_coarse1_v = get_data(test_data['u'],test_data['v'], i_indices_lf1, j_indices_lf1)    
    hf_coarse2_u, hf_coarse2_v = get_data(test_data['u'],test_data['v'], i_indices_lf2, j_indices_lf2)    
    hf_coarse3_u, hf_coarse3_v = get_data(test_data['u'],test_data['v'], i_indices_lf3, j_indices_lf3)    
    print(hf_coarse1_u.shape)
    err_coarse1 = u_test_snapshots_coarse1 - reshape_to_pod_2d_system_snapshots(hf_coarse1_u, hf_coarse1_v)
    err_coarse2 = u_test_snapshots_coarse2 - reshape_to_pod_2d_system_snapshots(hf_coarse2_u, hf_coarse2_v)
    err_coarse3 = u_test_snapshots_coarse3 - reshape_to_pod_2d_system_snapshots(hf_coarse3_u, hf_coarse3_v)
    plot_final_U_snapshot(np.abs(err_coarse1[:,:nt]), test_data_coarse1['x'], test_data_coarse1['y'], test_data_coarse1['x'].shape[0], n_steps=1001, save_path='./gif/error_lf1.pdf',vmin=0.0,vmax=1.0,title=r"$|U_{HF}-U_{LF}^{(3)}|$")
    plot_final_U_snapshot(np.abs(err_coarse2[:,:nt]), test_data_coarse2['x'], test_data_coarse2['y'], test_data_coarse2['x'].shape[0], n_steps=1001, save_path='./gif/error_lf2.pdf',vmin=0.0,vmax=1.0,title=r"$|U_{HF}-U_{LF}^{(2)}|$")
    plot_final_U_snapshot(np.abs(err_coarse3[:,:nt]), test_data_coarse3['x'], test_data_coarse3['y'], test_data_coarse3['x'].shape[0], n_steps=1001, save_path='./gif/error_lf3.pdf',vmin=0.0,vmax=1.0,title=r"$|U_{HF}-U_{LF}^{(1)}|$")

if __name__ == "__main__":
    main()
