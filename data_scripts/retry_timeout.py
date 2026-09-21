"""
Retry the files that previously timed out
"""

import sys
import time
import subprocess
import csv
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
SOLVER_ROOT = PROJECT_ROOT / "solvers"

TIMEOUT = 20 * 60

SOLVERS = {
    "AE_kissat": "AE_kissat2025_MAB/build/kissat",
    "cadical": "cadical-sc2025/build/cadical",
    "dynamic": "Dynamiccadical/dynamiccadical",
    "isasat": "IsaSAT/bin/isasat",
    "yalsat": "yalsat/yalsat",
}

# Load and save progress to a JSON file
def load_progress(progress_file):
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {}

def save_progress(progress, progress_file):
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

# Run solver on a CNF file
def run_solver(solver_path, cnf_path, time_limit):
    start = time.time()
    try:
        proc = subprocess.run(
            [str(solver_path), str(cnf_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=time_limit,
            text=True
        )
        elapsed = round(time.time() - start, 3)

        if proc.returncode == 10:
            return elapsed, "SAT"
        elif proc.returncode == 20:
            return elapsed, "UNSAT"
        return elapsed, f"ERR_{proc.returncode}"
    except subprocess.TimeoutExpired:
        if time_limit < TIMEOUT:
            return round(time.time() - start, 3), "CANCELLED"
        return round(time.time() - start, 3), "TIMEOUT"
    except Exception:
        return round(time.time() - start, 3), "ERROR"

# Write times to CSV times file
def write_times(times_row, times_file):
    fieldnames = ["file"]
    for name in SOLVERS:
        fieldnames.append(f"{name}_time")
        fieldnames.append(f"{name}_status")
    fieldnames.extend(["best_solver", "best_time"])
    
    write_header = not times_file.exists()
    with open(times_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(times_row)

# Process a CNF file with all solvers
def process_file(cnf_path, progress, progress_file, times_file):
    filename = cnf_path.name
    print(f"\n[ {filename} ]")
    
    solver_data = {}
    best_solver = None
    best_time = float('inf')
    any_success = False
    current_timeout = TIMEOUT
    early_stop = False
    
    for name, rel_path in SOLVERS.items():
        solver_path = SOLVER_ROOT / rel_path
        
        if early_stop:
            print(f" | {name}... skipped", end="", flush=True)
            solver_data[f"{name}_time"] = None
            solver_data[f"{name}_status"] = "SKIPPED"
            continue
        
        print(f" | {name}...", end="", flush=True)
        t, s = run_solver(solver_path, cnf_path, current_timeout)
        solver_data[f"{name}_time"] = t
        solver_data[f"{name}_status"] = s
        
        if name == "AE_kissat" and s == "TIMEOUT":
            early_stop = True
            print(" TIMEOUT", end="", flush=True)
            continue
        
        if s in ["SAT", "UNSAT"]:
            any_success = True
            if t < best_time:
                best_time = t
                best_solver = name
                current_timeout = min(current_timeout, best_time + 30)
        print(s, end="", flush=True)

    # record results and update progress
    times_row = {"file": filename, **solver_data,
                 "best_solver": best_solver if best_solver else "NONE",
                 "best_time": best_time if best_solver else None}
    write_times(times_row, times_file)
    
    status = "times_done" if any_success else "timeout"
    progress[filename] = status
    save_progress(progress, progress_file)
    
    return status


def main():
    parser = argparse.ArgumentParser(
        description="Retry timed-out CNF files with a longer timeout."
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to the dataset directory, e.g. data/track_main_2025"
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)
    
    dataset_name = data_dir.name
    output_dir = RESULTS_ROOT / dataset_name
    
    if not output_dir.exists():
        print(f"Error: results directory not found: {output_dir}")
        print("Run run_solvers.py first to generate progress.json.")
        sys.exit(1)
    
    progress_file = output_dir / "progress.json"
    times_file = output_dir / "times.csv"
    
    progress = load_progress(progress_file)
    
    timeout_files = [
        data_dir / f
        for f, status in progress.items()
        if status == "timeout" and f.endswith(".cnf")
    ]
    timeout_files = [f for f in timeout_files if f.exists()]
    
    if not timeout_files:
        print("No timed-out files found in progress.json.")
        return
    
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Retrying {len(timeout_files)} timed-out files with TIMEOUT={TIMEOUT}s")
    
    try:
        for idx, cnf_path in enumerate(timeout_files, 1):
            status = process_file(cnf_path, progress, progress_file, times_file)
            print(f" [{idx}/{len(timeout_files)}] -> {status}")
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")
    finally:
        print(f"\nResults appended to: {times_file}")

if __name__ == "__main__":
    main()