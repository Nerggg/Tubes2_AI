import numpy as np
import pandas as pd
import pickle
import concurrent.futures
from os import cpu_count
from tqdm import tqdm
import time

class KNN:
    '''
    KNN is a classification algorithm that uses the k-nearest neighbors algorithm.
    It is a lazy learning algorithm that stores all instances corresponding to training data in n-dimensional space.
    When an unknown discrete data is received, it analyzes the closest k number of instances saved (nearest neighbors) and returns the most common class as the prediction and for real-valued data it returns the mean of k nearest neighbors.

    Parameters
    ----------
    k : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    weights : {'uniform', 'distance'} or None, default='uniform'
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.

    p : int, default=2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric : str, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2.

    n_jobs : int, default=1
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Doesn't affect :meth:`fit` method.

    verbose : bool, default=True
        Whether to print progress messages or not.


    Methods
    -------
    fit(X_train, y_train)
        Fit the model using X_train as training data and y_train as target values.

    predict(X_test)
        Predict the class labels for the provided data.

    save(path)
        Save the model to the given path.

    load(path)
        Load a model from the given path.

    Examples
    --------
    >>> from src.lib.neighbors import KNN
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
    >>> model = KNN(k=5)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>>
    >>> print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    '''
    def __init__(self, k=5, n_jobs=1, metric='minkowski', p=2, weights='uniform', verbose=True):

        # Check for valid inputs
        if k < 1:
            raise ValueError("Invalid k. k must be greater than 0.")

        if not isinstance(k, int):
            raise ValueError("Invalid k. k must be an integer.")

        if metric not in ['manhattan', 'euclidean', 'minkowski']:
            raise ValueError("Invalid metric. Valid metrics are 'manhattan', 'euclidean' and 'minkowski'.")

        if not isinstance(metric, str):
            raise ValueError("Invalid metric. metric must be a string.")

        if p < 1:
            raise ValueError("Invalid p. p must be greater than 0.")

        if not isinstance(p, int) and not isinstance(p, float):
            raise ValueError("Invalid p. p must be a number.")

        if weights is None:
            self.weights = 'uniform'
        elif weights not in ['uniform', 'distance']:
            raise ValueError("Invalid weights. Valid values are 'uniform' and 'distance'.")

        if not isinstance(weights, str):
            raise ValueError("Invalid weights. weights must be a string.")

        if n_jobs < 1 and n_jobs != -1:
            raise ValueError("Invalid n_jobs. n_jobs must be greater than 0. Use -1 to use all available cores.")

        if not isinstance(n_jobs, int):
            raise ValueError("Invalid n_jobs. n_jobs must be an integer.")

        if not isinstance(verbose, bool):
            raise ValueError("Invalid verbose. verbose must be a boolean.")


        # Constructor
        self.k = k
        self.verbose = verbose
        self.metric = metric
        self.weights = weights

        # Distance metric
        if self.metric == 'manhattan':
            self.p = 1
        elif self.metric == 'euclidean':
            self.p = 2
        else:
            self.p = p

        # If n_jobs is -1, use all available cores
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

    def _get_nearest_neighbours(self, test):
        # Get nearest neighbours using Minkowski distance
        distances = np.linalg.norm(self.X_train - test, ord=self.p, axis=1)

        weights = None
        if self.weights == 'uniform':
            indices = np.argsort(distances)[:self.k]
        elif self.weights == 'distance':
            indices = np.argsort(distances)[:self.k]
            distances = distances[indices]
            weights = 1 / distances
            weights /= np.sum(weights)
        else:
            raise ValueError("Invalid weights. Valid values are 'uniform' and 'distance'.")

        return indices, weights

    def fit(self, X_train, y_train):
        # Save train data
        if isinstance(X_train, pd.DataFrame):
            if X_train.columns.empty:
                self.X_train = X_train.values.astype(float)
            else:
                self.X_train = X_train.iloc[:, :-1].values.astype(float)
        else:
            self.X_train = X_train.astype(float)

        self.y_train = y_train

    def _predict_instance(self, row):
        # Predict a single instance
        indices, weights = self._get_nearest_neighbours(row)
        labels = [self.y_train.iloc[neighbour] for neighbour in indices]

        if self.weights == 'uniform':
            prediction = max(set(labels), key=labels.count)
        elif self.weights == 'distance':
            weighted_labels = [weights[i] * labels[i] for i in range(len(labels))]
            prediction = max(set(weighted_labels), key=weighted_labels.count).astype(int)
        else:
            raise ValueError("Invalid weights. Valid values are 'uniform' and 'distance'.")

        return prediction

    def predict(self, X_test):
        if self.verbose:
            print(f"Using {self.n_jobs} {'core' if self.n_jobs == 1 else 'cores'} for predictions.")

        if isinstance(X_test, pd.DataFrame):
            if X_test.columns.empty:
                X_test = X_test.values.astype(float)
            else:
                X_test = X_test.iloc[:, :-1].values.astype(float)
        else:
            X_test = X_test.astype(float)

        start_time = time.time()

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            if self.verbose:
                results = list(tqdm(executor.map(self._predict_instance, X_test), total=len(X_test)))
            else:
                results = list(executor.map(self._predict_instance, X_test))

        elapsed_time = time.time() - start_time

        if self.verbose:
            print(f"Prediction completed in {elapsed_time:.2f} seconds.")

        return np.array(results)

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load(path):
        return pickle.load(open(path, 'rb'))