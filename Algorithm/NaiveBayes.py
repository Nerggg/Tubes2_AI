import numpy as np
from collections import defaultdict
import pickle

class NaiveBayes:
    """
    Naive Bayes classifier for Gaussian-distributed data.

    Attributes
    ----------
    class_probabilities : (dict)
        Probability of each class.

    mean : (dict)
        Mean of each feature for each class.

    variance : (dict)
        Variance of each feature for each class.

    Methods
    --------
    fit(X, y)
        Fit the model to training data.

    predict(X)
        Make predictions on new data.

    save(path)
        Save the trained model to a file.

    load(path)
        Load a trained model from a file.

    Examples
    --------
    >>> from src.lib.bayes import NaiveBayes
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import accuracy_score
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> iris = load_iris()
    >>> X = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> y = pd.Series(iris.target)
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>>
    >>> model = NaiveBayes()
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>>
    >>> print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    """
    def __init__(self):
        self.class_probabilities = None
        self.mean = None
        self.variance = None

    def fit(self, X, y):
        self.class_probabilities = self._calculate_class_probabilities(y)
        self.mean, self.variance = self._calculate_mean_and_variance(X, y)

    def _calculate_class_probabilities(self, y):
        class_counts = defaultdict(int)
        total_samples = len(y)

        for label in y:
            class_counts[label] += 1

        class_probabilities = {
            label: count / total_samples for label, count in class_counts.items()
        }
        return class_probabilities

    def _calculate_mean_and_variance(self, X, y):
        unique_classes = np.unique(y)
        mean = {}
        variance = {}

        for label in unique_classes:
            class_data = X[y == label]
            mean[label] = np.mean(class_data, axis=0)
            variance[label] = np.var(class_data, axis=0)

        return mean, variance

    def _gaussian_probability(self, x, mean, variance):
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
        return (1 / (np.sqrt(2 * np.pi * variance))) * exponent

    def _calculate_class_probabilities_given_features(self, features, label):
        class_probability = np.log(self.class_probabilities[label])

        for i, feature in enumerate(features):
            mean = self.mean[label][i]
            variance = self.variance[label][i]
            epsilon = 1e-10
            if variance < epsilon:
                variance = epsilon

            class_probability += np.log(
                self._gaussian_probability(feature, mean, variance)
            )

        return class_probability

    def predict(self, X):
        predictions = []

        for sample in X.values:
            class_probabilities = {
                label: self._calculate_class_probabilities_given_features(sample, label)
                for label in self.class_probabilities
            }

            predicted_class = max(class_probabilities, key=class_probabilities.get)
            predictions.append(predicted_class)

        return np.array(predictions)

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load(path):
        return pickle.load(open(path, 'rb'))