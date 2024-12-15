import numpy as np
import pandas as pd
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    def __init__(self):
        self.root = None

    def _entropy(self, labels):
        """Calculate the entropy of the dataset."""
        total = len(labels)
        label_counts = Counter(labels)
        entropy = -sum((count / total) * np.log2(count / total) for count in label_counts.values())
        logging.info(f"Entropy calculated: {entropy} for labels: {label_counts}")
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
            logging.info(f"Subset entropy for feature '{feature}' = {value}: {subset_entropy}")

        info_gain = overall_entropy - weighted_entropy
        logging.info(f"Information gain for feature '{feature}': {info_gain}")
        return info_gain

    def _choose_best_feature(self, data, features):
        """Select the feature with the highest information gain."""
        best_gain = float('-inf')
        best_feature = None

        logging.info("Choosing the best feature for split...")
        for feature in features:
            gain = self._information_gain(data, feature)
            logging.info(f"Feature '{feature}' has information gain: {gain}")
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        logging.info(f"Best feature selected: {best_feature}")
        return best_feature

    def _build_tree(self, data, features):
        """Recursively construct the decision tree."""
        labels = data.iloc[:, -1]

        # If all labels are identical, return a leaf node
        if len(labels.unique()) == 1:
            logging.info(f"All labels are the same: {labels.iloc[0]}. Creating a leaf node.")
            return TreeNode(output=labels.iloc[0])

        # If no features remain, return a leaf node with the majority label
        if not features:
            majority_label = labels.value_counts().idxmax()
            logging.info(f"No features left. Majority label: {majority_label}. Creating a leaf node.")
            return TreeNode(output=majority_label)

        # Select the feature with the highest information gain
        best_feature = self._choose_best_feature(data, features)
        logging.info(f"Splitting on feature: {best_feature}")

        # Create a new node for the selected feature
        node = TreeNode(feature=best_feature)

        # Remove the selected feature from the list of available features
        remaining_features = [f for f in features if f != best_feature]

        # Build child nodes for each value of the selected feature
        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value]
            logging.info(f"Creating subtree for feature '{best_feature}' = {value} with {len(subset)} samples.")
            node.branches[value] = self._build_tree(subset, remaining_features)

        return node

    def fit(self, X, y):
        """Train the ID3 decision tree model with features X and labels y."""
        logging.info("Starting training process...")
        data = pd.concat([X, y], axis=1)
        features = list(X.columns)
        self.root = self._build_tree(data, features)
        logging.info("Training completed.")

    def _predict_instance(self, instance, node):
        """Make a prediction for a single instance."""
        if node.is_leaf_node():
            logging.info(f"Reached leaf node. Prediction: {node.output}")
            return node.output

        feature_value = instance[node.feature]
        child_node = node.branches.get(feature_value)
        if child_node is None:
            logging.warning(f"Feature value '{feature_value}' not found in tree for feature '{node.feature}'.")
            return None  # Return None if the value is missing in the tree

        logging.info(f"Following branch: {node.feature} = {feature_value}")
        return self._predict_instance(instance, child_node)

    def predict(self, X):
        """Predict labels for all instances in the dataset X."""
        logging.info("Starting prediction process...")
        predictions = np.array([self._predict_instance(instance, self.root) for _, instance in X.iterrows()])
        logging.info("Prediction completed.")
        return predictions
