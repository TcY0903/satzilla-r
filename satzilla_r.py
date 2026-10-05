"""
SATZilla-R: predict the best solver for a CNF instance and run it.

Usage: python satzilla_r.py <cnf_file>
Example: python satzilla_r.py data/track_main_2025/instance.cnf
"""

import sys
import time
import pickle
import subprocess
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pysat.formula import CNF
from satzilla_features import extract_features

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "model_training"))
TRAINING_DIR = PROJECT_ROOT / "training_data"
MODEL_FILE = TRAINING_DIR / "pairwise_forests.pkl"
FEATURE_NAMES_FILE = PROJECT_ROOT / "data" / "feature_names.txt"
SOLVER_ROOT = PROJECT_ROOT / "solvers"

FEATURE_GROUPS = ["all"]
GROUP_TIMEOUT = 180
TIMEOUT = 10 * 60

SOLVERS = {
    "AE_kissat": "AE_kissat2025_MAB/build/kissat",
    "cadical": "cadical-sc2025/build/cadical",
    "dynamic": "Dynamiccadical/dynamiccadical",
    "isasat": "IsaSAT/bin/isasat",
    "yalsat": "yalsat/yalsat",
}

# Load the list of feature names
def get_feature_keys():
    if not FEATURE_NAMES_FILE.exists():
        print(f"Error: feature names file not found: {FEATURE_NAMES_FILE}")
        sys.exit(1)
    with open(FEATURE_NAMES_FILE, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# extract features from a CNF file
def extract_features_from_cnf(cnf_path):
    try:
        cnf = CNF(from_file=str(cnf_path))
        return extract_features(cnf, groups=FEATURE_GROUPS, group_timeout=GROUP_TIMEOUT)
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None

# Load the trained pairwise forests
def load_model():
    if not MODEL_FILE.exists():
        print(f"Error: model file not found: {MODEL_FILE}")
        print("Train the model first with train_model.py.")
        sys.exit(1)
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    return model

# Predict the best solver using pairwise voting
def vote(forests, x, solvers):
    votes = {s: 0 for s in solvers}
    x_2d = x.reshape(1, -1)
    pair_results = {}
    
    for (solver_a, solver_b), rf in forests.items():
        pred = rf.predict(x_2d)[0]
        winner = solver_a if pred == 1 else solver_b
        pair_results[(solver_a, solver_b)] = winner
        votes[winner] += 1
    
    max_votes = max(votes.values())
    top_solvers = [s for s, v in votes.items() if v == max_votes]
    
    if len(top_solvers) == 2:
        a, b = top_solvers
        key = (a, b) if (a, b) in pair_results else (b, a)
        if key in pair_results:
            return pair_results[key], votes
    
    return top_solvers[0], votes

def run_solver(solver_name, cnf_path):
    solver_path = SOLVER_ROOT / SOLVERS[solver_name]
    
    print(f"\nRunning {solver_name}...")
    start = time.time()
    try:
        proc = subprocess.run(
            [str(solver_path), str(cnf_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT,
            text=True
        )
        elapsed = round(time.time() - start, 3)
        if proc.returncode == 10:
            return "SAT", elapsed
        elif proc.returncode == 20:
            return "UNSAT", elapsed
        return f"ERR_{proc.returncode}", elapsed
    except subprocess.TimeoutExpired:
        return "TIMEOUT", round(time.time() - start, 3)
    except Exception as e:
        return f"ERROR: {e}", round(time.time() - start, 3)

def main():
    parser = argparse.ArgumentParser(
        description="SATZilla-R: predict the best solver for a CNF instance and run it."
    )
    parser.add_argument(
        "cnf_file",
        type=str,
        help="Path to a CNF file (relative to current working directory)."
    )
    args = parser.parse_args()
    
    cnf_path = Path(args.cnf_file).resolve()
    if not cnf_path.exists():
        print(f"Error: CNF file not found: {cnf_path}")
        sys.exit(1)
    
    print("--- SATZilla-R ---")
    print(f"Instance: {cnf_path.name}")
    
    # extract features
    print("\n[1/4] Extracting features...")
    features = extract_features_from_cnf(cnf_path)
    if features is None:
        print("  Feature extraction failed.")
        sys.exit(1)
    print(f"  Extracted {len(features)} features")
    
    # load model
    print("\n[2/4] Loading model...")
    model = load_model()
    forests = model["forests"]
    solvers = model["solvers"]
    feature_cols = model["feature_cols"]
    best_params = model["best_params"]
    print(f"  Model loaded (params: {best_params})")
    
    # predict
    print("\n[3/4] Predicting the best solver...")
    x = np.array([features.get(col, np.nan) for col in feature_cols], dtype=float)
    x = np.where(np.isinf(x), np.nan, x)
    best_solver, votes = vote(forests, x, solvers)
    
    print(f"  Votes: {votes}")
    print(f"  Predicted best solver: {best_solver}")
    
    # run the selected solver
    print("\n[4/4] Running the selected solver...")
    status, elapsed = run_solver(best_solver, cnf_path)
    print(f"  Result: {status} in {elapsed}s")

if __name__ == "__main__":
    main()