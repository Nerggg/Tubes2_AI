import numpy as np
import pandas as pd
from collections import Counter

class TreeNode:
    def __init__(self, feature=None, value=None, output=None):
        self.feature = feature  # Index of the feature used for splitting
        self.value = value  # Value of the feature used for the split
        self.output = output  # Predicted label if the node is a leaf
        self.branches = {}  # Dictionary for child nodes based on feature values

    def is_leaf_node(self):
        """Check if the current node is a leaf."""
        return self.output is not None

class SelfID3:
    def __init__(self, verbose=False):
        self.root = None
        self.verbose = verbose  # Parameter to control verbosity

    def _log(self, message):
        """Log message only if verbose is True."""
        if self.verbose:
            print(message)

    def _entropy(self, labels):
        """Calculate the entropy of the dataset."""
        total = len(labels)
        label_counts = Counter(labels)
        entropy = -sum((count / total) * np.log2(count / total) for count in label_counts.values())
        self._log(f"Entropy calculated: {entropy:.4f} for labels: {dict(label_counts)}")
        return entropy

    def _information_gain(self, data, feature):
        """Compute information gain for a specific feature."""
        overall_entropy = self._entropy(data.iloc[:, -1])
        values = data[feature].unique()

        weighted_entropy = 0
        for value in values:
            subset = data[data[feature] == value]
            subset_entropy = self._entropy(subset.iloc[:, -1])
            weighted_entropy += (len(subset) / len(data)) * subset_entropy
            self._log(f"Subset entropy for feature '{feature}' = {value}: {subset_entropy:.4f}")

        info_gain = overall_entropy - weighted_entropy
        self._log(f"Information gain for feature '{feature}': {info_gain:.4f}")
        return info_gain

    def _choose_best_feature(self, data, features):
        """Select the feature with the highest information gain."""
        best_gain = float('-inf')
        best_feature = None

        self._log("Choosing the best feature for split...")
        for feature in features:
            gain = self._information_gain(data, feature)
            self._log(f"Feature '{feature}' has information gain: {gain:.4f}")
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        self._log(f"Best feature selected: {best_feature}")
        return best_feature

    def _build_tree(self, data, features):
        """Recursively construct the decision tree."""
        labels = data.iloc[:, -1]

        # If all labels are identical, return a leaf node
        if len(labels.unique()) == 1:
            self._log(f"All labels are the same: {labels.iloc[0]}. Creating a leaf node.")
            return TreeNode(output=labels.iloc[0])

        # If no features remain, return a leaf node with the majority label
        if not features:
            majority_label = labels.value_counts().idxmax()
            self._log(f"No features left. Majority label: {majority_label}. Creating a leaf node.")
            return TreeNode(output=majority_label)

        # Select the feature with the highest information gain
        best_feature = self._choose_best_feature(data, features)
        self._log(f"Splitting on feature: {best_feature}")

        # Create a new node for the selected feature
        node = TreeNode(feature=best_feature)

        # Remove the selected feature from the list of available features
        remaining_features = [f for f in features if f != best_feature]

        # Build child nodes for each value of the selected feature
        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value]
            self._log(f"Creating subtree for feature '{best_feature}' = {value} with {len(subset)} samples.")
            node.branches[value] = self._build_tree(subset, remaining_features)

        return node

    def fit(self, X, y):
        """Train the ID3 decision tree model with features X and labels y."""
        self._log("Starting training process...")
        data = pd.concat([X, y], axis=1)
        features = list(X.columns)
        self.root = self._build_tree(data, features)
        self._log("Training completed.")

    def _predict_instance(self, instance, node):
        """Make a prediction for a single instance."""
        if node.is_leaf_node():
            self._log(f"Reached leaf node. Prediction: {node.output}")
            return node.output

        feature_value = instance[node.feature]
        child_node = node.branches.get(feature_value)
        if child_node is None:
            self._log(f"Feature value '{feature_value}' not found in tree for feature '{node.feature}'. Returning None.")
            return None  # Return None if the value is missing in the tree

        self._log(f"Following branch: {node.feature} = {feature_value}")
        return self._predict_instance(instance, child_node)

    def predict(self, X):
        """Predict labels for all instances in the dataset X."""
        self._log("Starting prediction process...")
        predictions = np.array([self._predict_instance(instance, self.root) for _, instance in X.iterrows()])
        self._log("Prediction completed.")
        return predictions
