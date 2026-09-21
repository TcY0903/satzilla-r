"""
Decision tree using gain ratio
"""

import numpy as np
from collections import Counter

# entropy
def entropy(y, weights=None):
    if len(y) == 0: return 0

    if weights is None:
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
    else:
        total = np.sum(weights)
        if total == 0:
            return 0
        probs = []
        for label in np.unique(y):
            w = np.sum(weights[y == label])
            probs.append(w / total)
        probs = np.array(probs)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

# information_gain
def information_gain(parent_y, left_y, right_y, left_w=None, right_w=None):
    n = len(parent_y)
    if n == 0: return 0

    if left_w is None:
        left_w = np.ones(len(left_y))
    if right_w is None:
        right_w = np.ones(len(right_y))
    
    total_w = np.sum(left_w) + np.sum(right_w)
    if total_w == 0: return 0
    weighted_entropy = (np.sum(left_w) / total_w) * entropy(left_y, left_w) + \
                       (np.sum(right_w) / total_w) * entropy(right_y, right_w)
    return entropy(parent_y) - weighted_entropy

# gain_ratio
def gain_ratio(parent_y, left_y, right_y, left_w=None, right_w=None):
    n = len(parent_y)
    if n == 0: return 0
    
    gain = information_gain(parent_y, left_y, right_y, left_w, right_w)
    
    if left_w is None:
        left_w = np.ones(len(left_y))
    if right_w is None:
        right_w = np.ones(len(right_y))
    
    total_w = np.sum(left_w) + np.sum(right_w)
    if total_w == 0: return 0
    p_left = np.sum(left_w) / total_w
    p_right = np.sum(right_w) / total_w

    split_info = 0
    if p_left > 0:
        split_info -= p_left * np.log2(p_left)
    if p_right > 0:
        split_info -= p_right * np.log2(p_right)
    if split_info == 0: return 0

    return gain / split_info

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None,
                 left_ratio=0.5, right_ratio=0.5):
        self.feature = feature          # index of the feature to split on
        self.threshold = threshold      # threshold value for the split
        self.left = left                # left child node
        self.right = right              # right child node
        self.value = value              # class label (if it's a leaf node)
        self.left_ratio = left_ratio    # proportion of samples going to the left child (for missing values)
        self.right_ratio = right_ratio  # proportion of samples going to the right child (for missing values)

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=5, n_features=None, random_state=None):
        self.max_depth = max_depth                      # maximum depth of the tree
        self.min_samples_split = min_samples_split      # minimum number of samples required to split an internal node
        self.n_features = n_features                    # number of features to consider when looking for the best split
        self.random_state = random_state                # random seed for reproducibility
        self.root = None                                # root node of the tree
        self.rng = np.random.RandomState(random_state)  # random number generator

    # fit the decision tree
    def fit(self, X, y):
        self.n_total_features = X.shape[1]
        if self.n_features is None:
            self.n_features = self.n_total_features
        weights = np.ones(len(y))
        self.root = self._build_tree(X, y, weights, depth=0)

    # build the decision tree recursively
    def _build_tree(self, X, y, weights, depth):
        n_samples = len(y)
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y, weights))

        # select a random subset of features
        feature_indices = self.rng.choice(
            self.n_total_features,
            size=min(self.n_features, self.n_total_features),
            replace=False
        )

        # collect feature-threshold-gain candidates
        candidates = []
        for feature in feature_indices:
            mask_not_nan = ~np.isnan(X[:, feature])
            if np.sum(mask_not_nan) < 2: continue   # skip features with less than 2 non-missing values
            
            values = X[mask_not_nan, feature]
            thresholds = np.unique(values)
            
            for threshold in thresholds:
                left_mask = mask_not_nan & (X[:, feature] <= threshold)
                right_mask = mask_not_nan & (X[:, feature] > threshold)
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0: continue
                
                gain = information_gain(y[mask_not_nan], y[left_mask], y[right_mask],
                    weights[left_mask], weights[right_mask])
                candidates.append((feature, threshold, gain, left_mask, right_mask))
        
        if not candidates:
            return Node(value=self._most_common_label(y, weights))
        
        # compute average gain
        avg_gain = np.mean([c[2] for c in candidates])
        
        # find the best feature and threshold based on gain ratio
        best_feature, best_threshold, best_ratio = None, None, -1
        for feature, threshold, gain, left_mask, right_mask in candidates:
            if gain < avg_gain: continue    # skip candidates with gain less than average gain
            
            mask_not_nan = ~np.isnan(X[:, feature])
            ratio = gain_ratio(y[mask_not_nan], y[left_mask], y[right_mask],
                weights[left_mask], weights[right_mask])
            if ratio > best_ratio:
                best_ratio = ratio
                best_feature = feature
                best_threshold = threshold

        if best_feature is None:
            return Node(value=self._most_common_label(y, weights))

        # Split the samples based on the best feature and threshold
        left_indices, left_weights = self._split_samples(
            X, best_feature, best_threshold, weights, go_left=True)
        right_indices, right_weights = self._split_samples(
            X, best_feature, best_threshold, weights, go_left=False)
        # Build the left and right subtrees
        left = self._build_tree(X[left_indices], y[left_indices], left_weights, depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], right_weights, depth + 1)

        # compute the proportion of samples going to left and right for missing values
        n_left = np.sum(left_weights)
        n_right = np.sum(right_weights)
        total = n_left + n_right
        left_ratio = n_left / total if total > 0 else 0.5
        right_ratio = n_right / total if total > 0 else 0.5

        return Node(feature=best_feature, threshold=best_threshold,
                    left=left, right=right,
                    left_ratio=left_ratio, right_ratio=right_ratio)

    # Split samples into left, right based on feature and threshold
    def _split_samples(self, X, feature, threshold, weights, go_left):
        mask_not_nan = ~np.isnan(X[:, feature])
        mask_nan = np.isnan(X[:, feature])

        # determine which samples go left or right based on the threshold
        if go_left:
            mask_value = mask_not_nan & (X[:, feature] <= threshold)
        else:
            mask_value = mask_not_nan & (X[:, feature] > threshold)
        indices = np.where(mask_value)[0]
        new_weights = weights[mask_value].copy()

        # handle missing values by distributing them according to the ratio of left/right samples
        if np.sum(mask_nan) > 0:
            n_left = np.sum(mask_not_nan & (X[:, feature] <= threshold))
            n_right = np.sum(mask_not_nan & (X[:, feature] > threshold))
            total = n_left + n_right
            
            if total > 0:
                ratio = n_left / total if go_left else n_right / total
                nan_indices = np.where(mask_nan)[0]
                indices = np.concatenate([indices, nan_indices])
                new_weights = np.concatenate([new_weights, weights[mask_nan] * ratio])
        
        return indices, new_weights

    # Get the most common (weighted) label
    def _most_common_label(self, y, weights=None):
        # weights is None, return the most common label
        if weights is None:
            return Counter(y).most_common(1)[0][0]

        # otherwise, return the label with the highest total weight
        label_weights = {}
        for label, w in zip(y, weights):
            label_weights[label] = label_weights.get(label, 0) + w
        return max(label_weights, key=label_weights.get)

    # Predict the labels for the given samples
    def predict(self, X):
        predictions = []
        for x in X:
            dist = self._predict_one(x, self.root)
            best_label = max(dist, key=dist.get)
            predictions.append(best_label)
        return np.array(predictions)

    def _predict_one(self, x, node):
        # leaf: return the value
        if node.value is not None:
            return {node.value: 1.0}

        # missing value: combine the predictions from left and right subtrees based on ratios
        if np.isnan(x[node.feature]):
            left_dist = self._predict_one(x, node.left)
            right_dist = self._predict_one(x, node.right)
            
            combined = {}
            for label, prob in left_dist.items():
                combined[label] = combined.get(label, 0) + prob * node.left_ratio
            for label, prob in right_dist.items():
                combined[label] = combined.get(label, 0) + prob * node.right_ratio
            return combined

        # normal case: go left or right based on the threshold
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)