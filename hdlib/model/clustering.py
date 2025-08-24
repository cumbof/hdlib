"""Clustering model.

It implements the __hdlib.model.clustering.ClusteringModel__ class object built according to the 
Hyperdimensional Computing (HDC) paradigm as described in _Gupta et al. 2022_ https://doi.org/10.1145/3503541."""

from typing import List, Optional

import numpy as np

from hdlib import Vector
from hdlib.arithmetic import bundle


class ClusteringModel:
    """Hyperdimensional ClusteringModel representation."""

    def __init__(
        self,
        k: int,
        n_features: int,
        size: int=10000,
        vtype: str="bipolar",
        max_iter: int=100,
        seed: Optional[int]=None
    ) -> "ClusteringModel":
        """Initialize a ClusteringModel object.

        Parameters
        ----------
        k : int
            Number of clusters.
        n_features : int
            Number of features.
        size : int, default 10000
            Vector dimensionality.
        vtype : str, deafult 'bipolar'
            Vector type.
        max_iter : int, default 100
            Maximum number of iterations.
        seed : int, default None
            Seed for reproducibility.

        Returns
        -------
        ClusteringModel
            The clustering model object.
        """

        if not isinstance(k, int) or k <= 0:
            raise ValueError("The number of clusters `k` must be a positive integer")

        if not isinstance(n_features, int) or n_features <= 0:
            raise ValueError("The number of features `n_features` must be a positive integer")

        if not isinstance(size, int) or size < 1000: # Reduced for faster testing
            raise ValueError("Vectors size must be an integer greater than or equal to 1000")

        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("The number of iterations `max_iter` must be a positive integer")

        if seed is not None and not isinstance(seed, int):
            raise TypeError("Seed must be an integer number")

        self.k = k
        self.n_features = n_features
        self.size = size
        self.vtype = vtype
        self.max_iter = max_iter
        self.seed = seed
        
        if self.seed is not None:
            np.random.seed(self.seed)

        # Internal projection matrix for encoding
        self.projection_matrix_ = np.random.randn(self.n_features, self.size)
        
        self.centroids_: List[Vector] = []
        self.labels_: np.ndarray = np.array([])

    def _encode(self, X: np.ndarray) -> List[Vector]:
        """Encodes a low-dimensional dataset into a list of hypervectors."""

        encoded_vectors = []

        for point in X:
            hd_vector_float = np.dot(point, self.projection_matrix_)
            hd_vector_bipolar = np.sign(hd_vector_float)
            encoded_vectors.append(Vector(vtype=self.vtype, vector=hd_vector_bipolar))

        return encoded_vectors

    def fit(self, X: np.ndarray) -> "ClusteringModel":
        """Compute HDC-based k-means clustering from a raw data matrix."""

        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise TypeError("Input `X` must be a 2D NumPy array.")

        if X.shape[1] != self.n_features:
            raise ValueError(f"Input data has {X.shape[1]} features, but model was initialized with {self.n_features}.")

        # Step 1: Encode the input data
        points = self._encode(X)

        num_points = len(points)
        self.labels_ = np.zeros(num_points, dtype=int)

        initial_indices = np.random.choice(num_points, size=self.k, replace=False)
        self.centroids_ = [Vector(vtype=points[i].vtype, vector=points[i].vector) for i in initial_indices]

        for i, centroid in enumerate(self.centroids_):
            centroid.name = f"centroid_{i}"

        for iteration in range(self.max_iter):
            new_labels = np.zeros(num_points, dtype=int)

            for i, point in enumerate(points):
                distances = [point.dist(centroid, method="cosine") for centroid in self.centroids_]
                new_labels[i] = np.argmin(distances)

            new_centroids = []

            for j in range(self.k):
                cluster_points = [points[i] for i, label in enumerate(new_labels) if label == j]

                if not cluster_points:
                    random_idx = np.random.randint(0, num_points)
                    new_centroids.append(Vector(vtype=points[random_idx].vtype, vector=points[random_idx].vector))

                else:
                    new_centroid_vector = cluster_points[0]
                    if len(cluster_points) > 1:
                        for point in cluster_points[1:]:
                            new_centroid_vector = bundle(new_centroid_vector, point)

                    new_centroid_vector.vector /= len(cluster_points)
                    new_centroids.append(new_centroid_vector)

            for i, centroid in enumerate(new_centroids):
                centroid.name = f"centroid_{i}"

            self.centroids_ = new_centroids

            if np.array_equal(self.labels_, new_labels):
                print(f"Converged after {iteration + 1} iterations.")
                break

            self.labels_ = new_labels
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the closest cluster for each sample in a raw data matrix."""

        if not self.centroids_:
            raise RuntimeError("The model has not been fitted yet. Call .fit() first.")

        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise TypeError("Input `X` must be a 2D NumPy array.")

        if X.shape[1] != self.n_features:
            raise ValueError(f"Input data has {X.shape[1]} features, but model was initialized with {self.n_features}.")

        # Encode the new data points
        points = self._encode(X)

        predictions = np.zeros(len(points), dtype=int)

        for i, point in enumerate(points):
            distances = [point.dist(centroid, method="cosine") for centroid in self.centroids_]
            predictions[i] = np.argmin(distances)

        return predictions
