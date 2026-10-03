"""
Evaluate the trained model on test set.
"""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training_data"
MODEL_FILE = TRAINING_DIR / "pairwise_forests.pkl"
TEST_FILE = TRAINING_DIR / "test.csv"

SOLVERS = ["AE_kissat", "cadical", "dynamic", "isasat", "yalsat"]

# Load the trained model
def load_model():
    if not MODEL_FILE.exists():
        print(f"Error: model file not found: {MODEL_FILE}")
        sys.exit(1)
    with open(MODEL_FILE, 'rb') as f:
        return pickle.load(f)

# Vote for the best solver based on predict model
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

# Evaluate the model on the test set
def evaluate_detailed(forests, df, feature_cols, title="Evaluation"):
    print("\n" + "--- " + title + " ---")
    
    y_true = []
    y_pred = []
    for _, row in df.iterrows():
        x = row[feature_cols].values.astype(float)
        x = np.where(np.isinf(x), np.nan, x)
        best_solver, _ = vote(forests, x, SOLVERS)
        y_true.append(row["best_solver"])
        y_pred.append(best_solver)

    all_solvers = sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    
    print(f"Overall accuracy: {correct / total:.4f} ({correct}/{total})")
    print(f"{'Solver':<12} {'Support':>8} {'Correct':>8} {'Accuracy':>10} {'Recall':>8}")
    print("-" * 60)
    for solver in all_solvers:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == solver and p == solver)
        support = sum(1 for t in y_true if t == solver)
        predicted_as = sum(1 for p in y_pred if p == solver)
        accuracy = tp / predicted_as if predicted_as > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        print(f"{solver:<12} {support:>8} {tp:>8} {accuracy:>10.4f} {recall:>8.4f}")
    
    print("\nMisclassification details:")
    for solver in all_solvers:
        errors = [(t, p) for t, p in zip(y_true, y_pred) if t == solver and t != p]
        if not errors: continue
        
        error_counts = {}
        for _, p in errors:
            error_counts[p] = error_counts.get(p, 0) + 1
        
        print(f"\n  True '{solver}' misclassified as:")
        for pred, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            pct = count / len(errors) * 100
            print(f"    -> {pred:<12} {count:>3} ({pct:5.1f}%)")
    
    return correct / total

def main():
    print("--- Evaluating the trained model ---")
    
    model = load_model()
    forests = model["forests"]
    solvers = model["solvers"]
    feature_cols = model["feature_cols"]
    best_params = model["best_params"]
    
    print(f"Model loaded (params: {best_params})")
    print(f"Solvers: {solvers}")
    
    if not TEST_FILE.exists():
        print(f"Error: test file not found: {TEST_FILE}")
        sys.exit(1)
    
    test_df = pd.read_csv(TEST_FILE)
    print(f"Test set: {len(test_df)} instances")
    
    evaluate_detailed(forests, test_df, feature_cols, "Final test evaluation")

if __name__ == "__main__":
    main()