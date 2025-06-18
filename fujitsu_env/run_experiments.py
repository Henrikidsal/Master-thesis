# run_experiments.py

import os
import subprocess

def run_experiment(periods, iterations, csv_file):
    """Constructs and runs the command for a single worker process."""
    command = [
        "python",
        "ucp_solver_worker.py",
        "--periods", str(periods),
        "--iterations", str(iterations),
        "--csv_file", csv_file
    ]
    
    print(f"\n>>>>>> Starting Worker: {' '.join(command)} <<<<<<\n")
    
    # Using subprocess.run will wait for the worker to complete
    # before starting the next one.
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the worker's output for monitoring
    print("--- Worker STDOUT ---")
    print(result.stdout)
    print("--- Worker STDERR ---")
    if result.stderr:
        print(result.stderr)
    print(f">>>>>> Worker for P={periods}, I={iterations} finished with exit code {result.returncode} <<<<<<")
    
    if result.returncode != 0:
        print("!!!!!! WARNING: Worker process exited with an error. !!!!!!")


if __name__ == '__main__':
    csv_file = "fujitsu_qubo_results_andreas_modified.csv"

    # Remove the results file at the very start of the entire experiment suite
    if os.path.exists(csv_file):
        os.remove(csv_file)
        print(f"Removed old results file: {csv_file}")

    # Define all the experiment configurations here
    run_configurations = [
        # (total_runs, iterations, periods)
        (50, 2_000, 3),
        (50, 5_000, 3),
        (50, 5_000, 5),
        (50, 5_000, 6)
    ]

    # Loop through each configuration
    for total_runs, iterations, periods in run_configurations:
        print(f"\n=====================================================================")
        print(f"--- Starting New Batch: {total_runs} runs with {iterations} iterations for {periods} time periods ---")
        print(f"=====================================================================")
        for i in range(total_runs):
            print(f"\n--- Master executing run {i+1}/{total_runs} for this batch ---")
            run_experiment(periods=periods, iterations=iterations, csv_file=csv_file)

    print("\nAll experiments have been completed.")