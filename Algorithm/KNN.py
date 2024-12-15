import numpy as np
import pandas as pd
import pickle
import concurrent.futures
from os import cpu_count
from tqdm import tqdm
import time

class SelfKNN:
    def __init__(self, k=5, n_jobs=1, metric='manhattan', p=1, weights='uniform', verbose=True):
        if not isinstance(k, int) or k < 1:
            raise ValueError("Parameter 'k' must be a positive integer.")

        if metric not in ['manhattan', 'euclidean', 'minkowski'] or not isinstance(metric, str):
            raise ValueError("Parameter 'metric' must be one of 'manhattan', 'euclidean', or 'minkowski'.")

        if not isinstance(p, (int, float)) or p < 1:
            raise ValueError("Parameter 'p' must be a positive number.")

        if weights not in [None, 'uniform', 'distance'] or (weights is not None and not isinstance(weights, str)):
            raise ValueError("Parameter 'weights' must be 'uniform', 'distance', or None.")

        if not isinstance(n_jobs, int) or (n_jobs < 1 and n_jobs != -1):
            raise ValueError("Parameter 'n_jobs' must be a positive integer or -1 to use all cores.")

        if not isinstance(verbose, bool):
            raise ValueError("Parameter 'verbose' must be a boolean.")

        self.k = k
        self.metric = metric
        self.verbose = verbose
        self.weights = weights if weights else 'uniform'
        self.p = p if metric == 'minkowski' else (1 if metric == 'manhattan' else 2)
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    def _compute_distances(self, test_point):
        if self.verbose:
            print(f"Calculating distances for test point: {test_point}")
        distances = np.linalg.norm(self.X_train - test_point, ord=self.p, axis=1)
        
        nearest_indices = np.argsort(distances)[:self.k]
        weight_factors = None

        if self.weights == 'distance':
            selected_distances = distances[nearest_indices]
            weight_factors = 1 / selected_distances
            weight_factors /= weight_factors.sum()

        return nearest_indices, weight_factors

    def fit(self, X_train, y_train):
        if self.verbose:
            print("Fitting the model with training data...")
        self.X_train = X_train.values.astype(float) if isinstance(X_train, pd.DataFrame) else X_train.astype(float)
        self.y_train = y_train
        if self.verbose:
            print("Model fitting complete.")

    def _predict_single_instance(self, instance):
        if self.verbose:
            print(f"Predicting for instance: {instance}")
        indices, weights = self._compute_distances(instance)
        neighbor_labels = [
            self.y_train.iloc[idx] if isinstance(self.y_train, pd.Series) else self.y_train[idx]
            for idx in indices
        ]

        if self.weights == 'uniform':
            prediction = max(set(neighbor_labels), key=neighbor_labels.count)
        elif self.weights == 'distance':
            label_weights = {}
            for i, label in enumerate(neighbor_labels):
                label_weights[label] = label_weights.get(label, 0) + weights[i]
            prediction = max(label_weights, key=label_weights.get)

        if self.verbose:
            print(f"Prediction for instance: {prediction}")
        return prediction

    def predict(self, X_test):
        if self.verbose:
            core_message = f"{self.n_jobs} core{'s' if self.n_jobs != 1 else ''}"
            print(f"Using {core_message} for prediction.")

        X_test = X_test.values.astype(float) if isinstance(X_test, pd.DataFrame) else X_test.astype(float)

        start_time = time.time()
        if self.verbose:
            print("Starting predictions...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(tqdm(executor.map(self._predict_single_instance, X_test), total=len(X_test))) if self.verbose else list(executor.map(self._predict_single_instance, X_test))

        if self.verbose:
            elapsed_time = time.time() - start_time
            print(f"Prediction completed in {elapsed_time:.2f} seconds.")

        return np.array(results)

    def save(self, file_path):
        if self.verbose:
            print(f"Saving model to {file_path}...")
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        if self.verbose:
            print("Model saved.")

    @staticmethod
    def load(file_path, verbose=False):
        if verbose:
            print(f"Loading model from {file_path}...")
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        if verbose:
            print("Model loaded.")
        return model