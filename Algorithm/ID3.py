import numpy as np
from collections import Counter
from treenode import TreeNode
import numpy as np
from collections import Counter

class TreeNode:
    def __init__(self, feature_idx=None, feature_val=None, prediction_probs=None, feature_importance=0):
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.feature_importance = feature_importance
        self.left = None
        self.right = None

    def node_def(self):
        """Return a string representation of the node."""
        if self.feature_idx is None:
            return f"Leaf Node: {self.prediction_probs}"
        return f"X[{self.feature_idx}] < {self.feature_val}, Importance: {self.feature_importance}"


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_leaf=1, min_information_gain=1e-7, feature_selection="all"):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.feature_selection = feature_selection
        self.tree = None

    def _entropy(self, probabilities):
        return -sum(p * np.log2(p) for p in probabilities if p > 0)

    def _class_probabilities(self, labels):
        total_count = len(labels)
        return [count / total_count for count in Counter(labels).values()]

    def _data_entropy(self, labels):
        return self._entropy(self._class_probabilities(labels))

    def _split(self, data, feature_idx, feature_val):
        left_mask = data[:, feature_idx] < feature_val
        return data[left_mask], data[~left_mask]

    def _partition_entropy(self, subsets):
        total_count = sum(len(subset) for subset in subsets)
        return sum(self._data_entropy(subset[:, -1]) * (len(subset) / total_count) for subset in subsets)

    def _find_best_split(self, data):
        num_features = data.shape[1] - 1
        best_split = None
        min_entropy = float('inf')

        features = self._select_features(num_features)
        for feature_idx in features:
            unique_vals = np.unique(data[:, feature_idx])
            split_points = (unique_vals[:-1] + unique_vals[1:]) / 2
            for val in split_points:
                left, right = self._split(data, feature_idx, val)
                if len(left) >= self.min_samples_leaf and len(right) >= self.min_samples_leaf:
                    entropy = self._partition_entropy([left, right])
                    if entropy < min_entropy:
                        min_entropy = entropy
                        best_split = (feature_idx, val, left, right)
        return best_split

    def _select_features(self, num_features):
        if self.feature_selection == "sqrt":
            return np.random.choice(num_features, int(np.sqrt(num_features)), replace=False)
        elif self.feature_selection == "log":
            return np.random.choice(num_features, int(np.log2(num_features)), replace=False)
        return np.arange(num_features)

    def _build_tree(self, data, depth):
        labels = data[:, -1]
        if depth >= self.max_depth or len(set(labels)) == 1:
            return TreeNode(prediction_probs=self._class_probabilities(labels))

        split = self._find_best_split(data)
        if not split:
            return TreeNode(prediction_probs=self._class_probabilities(labels))

        feature_idx, feature_val, left, right = split
        node = TreeNode(feature_idx, feature_val, self._class_probabilities(labels))
        node.left = self._build_tree(left, depth + 1)
        node.right = self._build_tree(right, depth + 1)
        return node

    def train(self, X, y):
        data = np.hstack((X, y.reshape(-1, 1)))
        self.tree = self._build_tree(data, depth=0)

    def _predict_one(self, x, node):
        while node:
            if node.feature_idx is None:
                return node.prediction_probs
            if x[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right
        return None

    def predict(self, X):
        return np.array([np.argmax(self._predict_one(x, self.tree)) for x in X])

    def predict_proba(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.tree
        if node:
            print("    " * level + "-> " + node.node_def())
            self.print_tree(node.left, level + 1)
            self.print_tree(node.right, level + 1)
