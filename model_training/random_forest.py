"""
Random Forest using DecisionTree
"""

import numpy as np
from collections import Counter
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=5,
                 n_features=None, random_state=None):
        self.n_trees = n_trees                              # number of trees in the forest
        self.max_depth = max_depth                          # maximum depth of each tree
        self.min_samples_split = min_samples_split          # minimum number of samples required to split an internal node
        self.n_features = n_features                        # number of features to consider when looking for the best split
        self.random_state = random_state                    # random seed for reproducibility
        self.trees = []                                     # list to hold all decision trees built in the forest
        self.rng = np.random.RandomState(random_state)      # random number generator

    # Fit the random forest model
    def fit(self, X, y):
        n_samples, n_features_total = X.shape
        
        if self.n_features is None:
            n_features_per_tree = max(1, int(np.sqrt(n_features_total)))
        else:
            n_features_per_tree = self.n_features

        self.trees = []
        for i in range(self.n_trees):
            # Bootstrap sampling: randomly sample with replacement from the training data
            indices = self.rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # train a decision tree on the bootstrap sample
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=n_features_per_tree,
                random_state=self.rng.randint(0, 100000)
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    # Predict best labels for the input samples
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])