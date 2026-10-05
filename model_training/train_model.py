"""
Train pairwise random forests
"""

import pickle
import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from random_forest import RandomForest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training_data"
TRAIN_PAIRWISE_DIR = TRAINING_DIR / "pairwise_train"
VAL_PAIRWISE_DIR = TRAINING_DIR / "pairwise_validation"
VAL_FILE = TRAINING_DIR / "validation.csv"
TEST_FILE = TRAINING_DIR / "test.csv"
MODEL_FILE = TRAINING_DIR / "pairwise_forests.pkl"

SOLVERS = ["AE_kissat", "cadical", "dynamic", "isasat", "yalsat"]
RANDOM_SEED = 42

PARAM_GRID = [
    {"n_trees": 15, "max_depth": 12, "min_samples_split": 5},
    {"n_trees": 25, "max_depth": 10, "min_samples_split": 8},
    {"n_trees": 35, "max_depth": 8, "min_samples_split": 10},
]


# Load pairwise training/validation data for a given solver pair
def load_pairwise(pair_name, split="train"):
    if split == "train":
        file = TRAIN_PAIRWISE_DIR / f"{pair_name}.csv"
    else:
        file = VAL_PAIRWISE_DIR / f"{pair_name}.csv"
    if not file.exists(): return None

    df = pd.read_csv(file)
    if "file" in df.columns:
        df = df.drop(columns=["file"])
    X = df.drop(columns=["label"]).values.astype(float)
    y = df["label"].values
    X = np.where(np.isinf(X), np.nan, X)
    return X, y

# Train random forest for all solver pairs
def train_all_pairs(params):
    forests = {}
    for solver_a, solver_b in combinations(SOLVERS, 2):
        pair_name = f"{solver_a}_vs_{solver_b}"
        result = load_pairwise(pair_name, split="train")
        if result is None: continue

        X, y = result
        rf = RandomForest(
            n_trees=params["n_trees"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=RANDOM_SEED
        )
        rf.fit(X, y)
        forests[(solver_a, solver_b)] = rf
    return forests

# Predict the best solver for an instance using voting
def vote(forests, x):
    votes = {s: 0 for s in SOLVERS}
    x_2d = x.reshape(1, -1)
    pair_results = {}

    # vote for winner in each pairwise comparison
    for (solver_a, solver_b), rf in forests.items():
        pred = rf.predict(x_2d)[0]
        winner = solver_a if pred == 1 else solver_b
        pair_results[(solver_a, solver_b)] = winner
        votes[winner] += 1
    
    max_votes = max(votes.values())
    top_solvers = [s for s, v in votes.items() if v == max_votes]

    # if there is a tie, use the pairwise classifier between the tied solvers to decide
    if len(top_solvers) == 2:
        a, b = top_solvers
        key = (a, b) if (a, b) in pair_results else (b, a)
        if key in pair_results:
            return pair_results[key], votes
    
    return top_solvers[0], votes

# Evaluate voting accuracy on a dataset
def evaluate(forests, df, feature_cols, title="Evaluation", detailed=False):
    y_true, y_pred = [], []
    
    # for each instance, predict the best solver and compare with true label
    for _, row in df.iterrows():
        x = row[feature_cols].values.astype(float)
        x = np.where(np.isinf(x), np.nan, x)
        best_solver, _ = vote(forests, x)
        y_true.append(row["best_solver"])
        y_pred.append(best_solver)

    acc = accuracy_score(y_true, y_pred)
    correct = sum(np.array(y_true) == np.array(y_pred))
    print(f"\n{title}: accuracy = {acc:.4f} ({correct}/{len(y_true)})")
    
    if detailed:
        print("\nPer-class metrics:")
        print(classification_report(y_true, y_pred, zero_division=0))

    return acc


def main():
    print("--- Training pairwise Random Forests ---")

    # Load validation set and feature columns
    val_df = pd.read_csv(VAL_FILE)
    feature_cols = [c for c in val_df.columns 
                    if c != "file" and c != "best_solver"
                    and not c.endswith("_time") and not c.endswith("_status")]
    
    # train random forests, evaluate on validation set to find best hyperparameters
    if len(PARAM_GRID) == 1:
        # only one hyperparameter set, no need to search
        best_params = PARAM_GRID[0]
        best_val_acc = None
        print(f"\nOnly one parameter set found: {best_params}")
        print("Skipping validation search.")
    else:
        best_params, best_val_acc = None, -1
        for params in PARAM_GRID:
            print(f"\nParams: {params}")
            forests = train_all_pairs(params)
            val_acc = evaluate(forests, val_df, feature_cols, "Validation")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = params
        print(f"\nBest params: {best_params} (val acc: {best_val_acc:.4f})")
    
    # retrain on train + validation set using best hyperparameters
    print("\nRetraining on train + validation...")
    forests = {}
    for solver_a, solver_b in combinations(SOLVERS, 2):
        pair_name = f"{solver_a}_vs_{solver_b}"
        train_result = load_pairwise(pair_name, split="train")
        val_result = load_pairwise(pair_name, split="validation")
        
        if train_result is None or val_result is None:
            continue
        
        X_train, y_train = train_result
        X_val, y_val = val_result
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        
        # train a random forest on the combined dataset
        rf = RandomForest(
            n_trees=best_params["n_trees"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            random_state=RANDOM_SEED
        )
        rf.fit(X_combined, y_combined)
        forests[(solver_a, solver_b)] = rf
    
    # save the trained forests and best hyperparameters to a file
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            "forests": forests,
            "best_params": best_params,
            "solvers": SOLVERS,
            "feature_cols": feature_cols
        }, f)
    print(f"\nModel saved to {MODEL_FILE}")
    
    # evaluate on the final test set
    if TEST_FILE.exists():
        test_df = pd.read_csv(TEST_FILE)
        evaluate(forests, test_df, feature_cols, "Final test evaluation", detailed=True)

if __name__ == "__main__":
    main()