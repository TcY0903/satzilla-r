"""
Extract features from files that have successfully completed solver runs.
"""

import sys
import csv
import json
import argparse
from pathlib import Path
from pysat.formula import CNF
from satzilla_features import extract_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
FEATURE_NAMES_FILE = PROJECT_ROOT / "data" / "feature_names.txt"

FEATURE_GROUPS = ["all"]
GROUP_TIMEOUT = 120

# Load and save progress to a JSON file
def load_progress(progress_file):
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {}

def save_progress(progress, progress_file):
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

# Extract features from a CNF file
def extract_features_file(cnf_path):
    try:
        cnf = CNF(from_file=str(cnf_path))
        return extract_features(cnf, groups=FEATURE_GROUPS, group_timeout=GROUP_TIMEOUT)
    except Exception as e:
        return {"error": str(e)}

# Get the list of feature keys
def get_feature_keys():
    # feature_names.txt doesn't exist: extract features from the first available CNF file
    if not FEATURE_NAMES_FILE.exists():
        print(f"feature_names.txt not found at {FEATURE_NAMES_FILE}")
        print("Extracting from first available CNF file...")
        
        for data_dir in (PROJECT_ROOT / "data").iterdir():
            if data_dir.is_dir() and data_dir.name != "results":
                cnf_files = sorted([f for f in data_dir.iterdir() if f.suffix == ".cnf"])
                if cnf_files:
                    sample = extract_features_file(cnf_files[0])
                    feature_keys = sorted([k for k in sample.keys() if k != "error"])
                    
                    FEATURE_NAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
                    with open(FEATURE_NAMES_FILE, 'w') as f:
                        for name in feature_keys:
                            f.write(f"{name}\n")
                    print(f"Saved {len(feature_keys)} feature names to {FEATURE_NAMES_FILE}")
                    return feature_keys
        raise FileNotFoundError("No CNF files found to extract feature names.")

    # feature_names.txt exists: load feature names from the file
    with open(FEATURE_NAMES_FILE, 'r') as f:
        feature_keys = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(feature_keys)} feature names from {FEATURE_NAMES_FILE}")
    return feature_keys

# Write features to CSV features file
def write_features(features, filename, feature_keys, features_file):
    write_header = not features_file.exists()
    with open(features_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["file"] + feature_keys)
        if write_header:
            writer.writeheader()
        writer.writerow({"file": filename, **features})


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from solved CNF files."
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
    features_file = output_dir / "features.csv"
    
    progress = load_progress(progress_file)
    
    # process files that have been solved (status "times_done")
    solved_files = [
        data_dir / f
        for f, status in progress.items()
        if status == "times_done" and f.endswith(".cnf")
    ]
    
    solved_files = [f for f in solved_files if f.exists()]
    if not solved_files:
        print("No solved files found in progress.json.")
        return
    
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Found {len(solved_files)} solved files")
    
    # check existing features to avoid re-extracting
    existing_features = set()
    if features_file.exists():
        with open(features_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    existing_features.add(row[0])
    
    pending = [f for f in solved_files if f.name not in existing_features]
    print(f"Pending: {len(pending)}")
    if not pending:
        print("All features already extracted.")
        return
    
    feature_keys = get_feature_keys()
    
    try:
        for idx, cnf_path in enumerate(pending, 1):
            filename = cnf_path.name
            print(f"\n[ {filename} ]")
            
            print("  feature...", end="", flush=True)
            features = extract_features_file(cnf_path)
            if "error" in features:
                print(f" FAIL: {features['error']}")
                continue
            print("done", end="")
            
            write_features(features, filename, feature_keys, features_file)
            print(" | saved", end="")
            print(f" [{idx}/{len(pending)}]")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print(f"\nFeatures saved to: {features_file}")

if __name__ == "__main__":
    main()