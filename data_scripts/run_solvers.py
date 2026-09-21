"""
Run solvers on CNF files and record times.
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

SOLVERS = {
    "AE_kissat": "AE_kissat2025_MAB/build/kissat",
    "cadical": "cadical-sc2025/build/cadical",
    "dynamic": "Dynamiccadical/dynamiccadical",
    "isasat": "IsaSAT/bin/isasat",
    "yalsat": "yalsat/yalsat",
}
TIMEOUT = 5 * 60

# Get all CNF files in the directory
def get_cnf_files(data_dir):
    return sorted([f for f in data_dir.iterdir() if f.suffix == ".cnf"])

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

        # if AE_kissat (first solver) times out, skip remaining solvers
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

    # Record results and update progress
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
        description="Run SAT solvers on a dataset of CNF files."
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
    output_dir.mkdir(parents=True, exist_ok=True)
    
    progress_file = output_dir / "progress.json"
    times_file = output_dir / "times.csv"
    
    cnf_files = get_cnf_files(data_dir)
    progress = load_progress(progress_file)
    
    pending = [f for f in cnf_files 
               if progress.get(f.name) not in ["times_done", "timeout"]]
    
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Total: {len(cnf_files)} | Pending: {len(pending)}")
    
    if not pending:
        print("All files already processed.")
        return
    
    try:
        for idx, cnf_path in enumerate(pending, 1):
            status = process_file(cnf_path, progress, progress_file, times_file)
            print(f" [{idx}/{len(pending)}] -> {status}")
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")
    finally:
        print(f"\nTimes saved to: {times_file}")

if __name__ == "__main__":
    main()