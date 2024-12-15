import numpy as np
from collections import defaultdict
import pickle

class NaiveBayes:
    def __init__(self, verbose=False):
        self.class_probabilities = None
        self.mean = None
        self.variance = None
        self.verbose = verbose

    def fit(self, X, y):
        if self.verbose:
            print("Fitting model...")
        self.class_probabilities = self._calculate_class_probabilities(y)
        self.mean, self.variance = self._calculate_mean_and_variance(X, y)
        if self.verbose:
            print("Model fitting complete.")

    def _calculate_class_probabilities(self, y):
        if self.verbose:
            print("Calculating class probabilities...")
        class_counts = defaultdict(int)
        total_samples = len(y)

        for label in y:
            class_counts[label] += 1

        class_probabilities = {label: count / total_samples for label, count in class_counts.items()}
        if self.verbose:
            print(f"Class probabilities: {class_probabilities}")
        return class_probabilities

    def _calculate_mean_and_variance(self, X, y):
        if self.verbose:
            print("Calculating mean and variance for each class...")
        unique_classes = np.unique(y)
        mean = {}
        variance = {}

        for label in unique_classes:
            class_data = X[y == label]
            mean[label] = np.mean(class_data, axis=0)
            variance[label] = np.var(class_data, axis=0)
            if self.verbose:
                print(f"Class {label}: Mean = {mean[label]}, Variance = {variance[label]}")

        return mean, variance

    def _gaussian_probability(self, x, mean, variance):
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
        return (1 / np.sqrt(2 * np.pi * variance)) * exponent

    def _calculate_class_likelihood(self, features, label):
        if self.verbose:
            print(f"Calculating likelihood for class {label}...")
        log_class_probability = np.log(self.class_probabilities[label])

        for i, feature in enumerate(features):
            mean = self.mean[label][i]
            variance = self.variance[label][i]
            epsilon = 1e-10 
            if variance < epsilon:
                variance = epsilon

            log_class_probability += np.log(self._gaussian_probability(feature, mean, variance))

        if self.verbose:
            print(f"Class {label} likelihood: {log_class_probability}")
        return log_class_probability

    def predict(self, X):
        if self.verbose:
            print("Making predictions...")
        predictions = []

        for sample in X.values:
            class_likelihoods = {
                label: self._calculate_class_likelihood(sample, label)
                for label in self.class_probabilities
            }

            predicted_class = max(class_likelihoods, key=class_likelihoods.get)
            if self.verbose:
                print(f"Sample: {sample}, Predicted class: {predicted_class}")
            predictions.append(predicted_class)

        if self.verbose:
            print("Prediction complete.")
        return np.array(predictions)

    def save(self, path):
        if self.verbose:
            print(f"Saving model to {path}...")
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        if self.verbose:
            print("Model saved.")

    @staticmethod
    def load(path, verbose=False):
        if verbose:
            print(f"Loading model from {path}...")
        with open(path, 'rb') as file:
            model = pickle.load(file)
        if verbose:
            print("Model loaded.")
        return model