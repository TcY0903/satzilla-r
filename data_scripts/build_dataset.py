"""
Build full dataset with train/validation/test split,
Generate pairwise comparison datasets from the training set.
"""

import sys
import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
TRAINING_DIR = PROJECT_ROOT / "training_data"

SOLVERS = ["AE_kissat", "cadical", "dynamic", "isasat", "yalsat"]

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
RANDOM_SEED = 42

# Find all result folders with features.csv and times.csv
def find_result_folders():
    folders = []
    for folder in RESULTS_ROOT.iterdir():
        if not folder.is_dir(): continue
        features_file = folder / "features.csv"
        times_file = folder / "times.csv"
        if features_file.exists() and times_file.exists():
            folders.append(folder)
    return sorted(folders)

# Load and merge features and times from all folders
def load_and_merge(folders):
    all_dfs = []
    
    for folder in folders:
        features_file = folder / "features.csv"
        times_file = folder / "times.csv"
        
        features = pd.read_csv(features_file)
        times = pd.read_csv(times_file)
        
        df = features.merge(times, on="file")
        df["file"] = folder.name + "/" + df["file"]
        
        all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError("No valid datasets found.")
    
    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"Total merged rows: {len(merged)}")
    return merged

# Filter out rows where no solver succeeded
def filter_successful(df):
    before = len(df)
    df = df[df["best_solver"] != "NONE"].copy()
    print(f"Filtered out {before - len(df)} rows with no successful solver.")
    return df

# Print distribution
def print_distribution(df, title="Distribution"):
    print("\n--- " + title + " ---")
    total = len(df)
    counts = df["best_solver"].value_counts()
    for solver, count in counts.items():
        pct = count / total * 100
        print(f"  {solver:12s} {count:4d} ({pct:5.1f}%)")
    print(f"Total: {total}")

# Build pairwise dataset for a pair of solvers
def build_pairwise_dataset(df, solver_a, solver_b):
    status_a = df[f"{solver_a}_status"]
    status_b = df[f"{solver_b}_status"]
    time_a = df[f"{solver_a}_time"]
    time_b = df[f"{solver_b}_time"]
    
    valid_a = status_a.isin(["SAT", "UNSAT"])
    valid_b = status_b.isin(["SAT", "UNSAT"])
    valid_mask = valid_a | valid_b

    # Assign labels: 1 if solver_a is faster, 0 if solver_b is faster
    labels = np.where(
        valid_a & valid_b,
        (time_a < time_b).astype(int),
        np.where(valid_a, 1, 0)
    )
    
    feature_cols = [c for c in df.columns 
                    if c != "file" 
                    and c != "best_solver"
                    and not c.endswith("_time") 
                    and not c.endswith("_status")]
    
    result = df.loc[valid_mask, ["file"] + feature_cols].copy()
    result["label"] = labels[valid_mask]
    return result

# Generate pairwise datasets for a given split
def generate_pairwise_for_split(df, split_name):
    split_dir = TRAINING_DIR / f"pairwise_{split_name}"
    split_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating pairwise datasets for {split_name}...")
    for solver_a, solver_b in combinations(SOLVERS, 2):
        result = build_pairwise_dataset(df, solver_a, solver_b)
        pair_name = f"{solver_a}_vs_{solver_b}"
        output_file = split_dir / f"{pair_name}.csv"
        result.to_csv(output_file, index=False)
        
        n_a = (result["label"] == 1).sum()
        n_b = (result["label"] == 0).sum()
        print(f"  {pair_name}: {len(result)} samples ({n_a} {solver_a}, {n_b} {solver_b})")


def main():
    print("--- Building full dataset and pairwise datasets ---")
    
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    
    # find all result folders with features.csv and times.csv
    folders = find_result_folders()
    if not folders:
        print(f"No valid result folders found in {RESULTS_ROOT}")
        sys.exit(1)
    
    print(f"Found {len(folders)} result folders:")
    for f in folders:
        print(f"  - {f.name}")
    
    merged = load_and_merge(folders)
    merged = filter_successful(merged)
    merged = merged.sort_values(by="file").reset_index(drop=True)
    print_distribution(merged, "Full dataset distribution")
    
    # split train/validation/test
    train_df, temp_df = train_test_split(
        merged, test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED, stratify=merged["best_solver"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED, stratify=temp_df["best_solver"]
    )
    
    train_df.to_csv(TRAINING_DIR / "train.csv", index=False)
    val_df.to_csv(TRAINING_DIR / "validation.csv", index=False)
    test_df.to_csv(TRAINING_DIR / "test.csv", index=False)
    
    print(f"\nSaved:")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Validation: {len(val_df)} rows")
    print(f"  Test: {len(test_df)} rows")
    
    # Generate pairwise datasets for train and validation
    generate_pairwise_for_split(train_df, "train")
    generate_pairwise_for_split(val_df, "validation")

if __name__ == "__main__":
    main()