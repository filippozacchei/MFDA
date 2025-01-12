import os
import numpy as np
from itertools import product
from timeit import repeat
from matplotlib import pyplot as plt
from model import Model  # Assuming your custom solver class is in 'model'

# Environment Setup
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Global Configuration
np.random.seed(123)

# Simulation Parameters
RESOLUTIONS = [(60, 60), (20, 20), (15,15), (12, 12), (10, 10)]
FIELD_MEAN = 1
FIELD_STD_DEV = 1
LAMB_COV = 0.1
MKL_VALUES = [64, 64, 64, 64, 64]
NUM_DATAPOINTS = 64

# Set up data grid for evaluation points
x_data = y_data = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
DATAPOINTS = np.array(list(product(x_data, y_data)))


def create_solvers() -> dict:
    """
    Create and configure the solvers based on predefined resolutions and parameters.
    Returns a dictionary of solver instances.
    """
    print("Creating solvers with different resolutions...\n")
    return {
        "h1": Model(RESOLUTIONS[0], FIELD_MEAN, FIELD_STD_DEV, MKL_VALUES[0], LAMB_COV),
        "h2": Model(RESOLUTIONS[1], FIELD_MEAN, FIELD_STD_DEV, MKL_VALUES[1], LAMB_COV),
        "h3": Model(RESOLUTIONS[2], FIELD_MEAN, FIELD_STD_DEV, MKL_VALUES[2], LAMB_COV),
        "h4": Model(RESOLUTIONS[3], FIELD_MEAN, FIELD_STD_DEV, MKL_VALUES[3], LAMB_COV),
        "h5": Model(RESOLUTIONS[4], FIELD_MEAN, FIELD_STD_DEV, MKL_VALUES[4], LAMB_COV),
    }


def synchronize_transmissivity(solver_high: Model, solver_low: Model) -> None:
    """
    Synchronize the transmissivity field between a higher fidelity solver
    and a lower fidelity one by transferring eigenvalues and eigenvectors.
    """
    print(f"Synchronizing transmissivity between high and low fidelity solvers...\n")
    high_coords = solver_high.solver.mesh.coordinates()
    low_coords = solver_low.solver.mesh.coordinates()
    
    # Define structured array for row-wise comparison
    dtype = {'names': [f'f{i}' for i in range(high_coords.shape[1])], 'formats': [high_coords.dtype] * high_coords.shape[1]}
    
    # Convert to structured arrays for comparison
    high_structured = np.array(high_coords).view(dtype)
    low_structured = np.array(low_coords).view(dtype)
    
    # Mask for matching coordinates
    bool_mask = np.in1d(high_structured, low_structured)
    
    # Transfer eigenvalues and corresponding eigenvectors
    solver_low.random_process.eigenvalues = solver_high.random_process.eigenvalues
    solver_low.random_process.eigenvectors = solver_high.random_process.eigenvectors[bool_mask]


def run_solver(solver: Model, input_data: np.ndarray, datapoints: np.ndarray) -> np.ndarray:
    """
    Run the solver for the given input data and datapoints.
    Returns the solver's computed data.
    """
    solver.solve(input_data)
    return solver.get_data(datapoints)


def time_solver_execution(solver: Model, input_data: np.ndarray, datapoints: np.ndarray, repetitions: int = 500) -> float:
    """
    Times the execution of the solver using `timeit.repeat` and returns
    the minimum execution time over the specified number of repetitions.
    """
    print(f"Timing execution for solver: {solver}...\n")
    return np.min(repeat(lambda: run_solver(solver, input_data, datapoints), number=1, repeat=repetitions))


def print_results(execution_times: dict, speedups: dict) -> None:
    """
    Prints the execution times and speedup factors for the solvers.
    """
    print("\n----- Execution Times -----")
    for solver_key, time in execution_times.items():
        print(f"Average time per run for {solver_key}: {time:.6f} seconds")
    
    print("\n----- Speedup Coefficients (Relative to h1) -----")
    for solver_key, speedup in speedups.items():
        print(f"{solver_key}: {speedup:.2f}x speedup")


def save_solver_plot(solver: Model, filename: str, limits: list = [-2, 4.5]) -> None:
    """
    Saves the plot of the solver's results within the given limits to a file.
    """
    solver.plot(limits=limits)
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {filename}\n")


def main() -> None:
    """
    Main function to create solvers, synchronize transmissivity, measure execution times,
    calculate speedups, and output the results.
    """
    print("Starting simulation...\n")

    # Create solvers
    solvers = create_solvers()
    
    # Synchronize solvers' transmissivity fields
    synchronize_transmissivity(solvers['h1'], solvers['h2'])
    synchronize_transmissivity(solvers['h1'], solvers['h3'])
    synchronize_transmissivity(solvers['h1'], solvers['h4'])
    synchronize_transmissivity(solvers['h1'], solvers['h5'])

    # Input data for solvers
    input_data = np.ones(NUM_DATAPOINTS)

    # Measure execution times for each solver
    execution_times = {
        'h1': time_solver_execution(solvers['h1'], input_data, DATAPOINTS),
        'h2': time_solver_execution(solvers['h2'], input_data, DATAPOINTS),
        'h3': time_solver_execution(solvers['h3'], input_data, DATAPOINTS),
        'h4': time_solver_execution(solvers['h4'], input_data, DATAPOINTS),
        'h5': time_solver_execution(solvers['h5'], input_data, DATAPOINTS),
    }

    # Calculate speedups relative to h1
    speedups = {
        'h2': execution_times['h1'] / execution_times['h2'],
        'h3': execution_times['h1'] / execution_times['h3'],
        'h4': execution_times['h1'] / execution_times['h4'],
        'h5': execution_times['h1'] / execution_times['h5'] 
    }

    # Print execution times and speedup results
    print_results(execution_times, speedups)

    # Save plots for each solver
    save_solver_plot(solvers['h1'], '../images/solver_h1_60.png')
    save_solver_plot(solvers['h2'], '../images/solver_h2_20.png')
    save_solver_plot(solvers['h3'], '../images/solver_h3_15.png')
    save_solver_plot(solvers['h4'], '../images/solver_h4_12.png')
    save_solver_plot(solvers['h5'], '../images/solver_h4_11.png')


# Entry point
if __name__ == "__main__":
    main()