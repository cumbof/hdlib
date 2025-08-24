"""Similarity-preserving encoding and multi-model regression.

It implements the __hdlib.model.regression.RegressionEncoder__ and __hdlib.model.regression.RegressionModel__ class objects 
built according to the Hyperdimensional Computing (HDC) paradigm as described in _Hernández-Cano et al. 2021_ https://doi.org/10.1109/DAC18074.2021.9586284."""

import numpy as np
from scipy.special import softmax

from hdlib import Vector


class RegressionEncoder:
    """Implements the similarity-preserving encoding from the RegHD article.
    Maps input features to high-dimensional real-valued hypervectors.

    The encoding function is h_i = cos(F_bar . B_i + b_i) * sin(F_bar . B_i), where F_bar is the input feature vector, 
    B_i are random base hypervectors, and b_i are random biases.
    """

    def __init__(self, D: int, n_features: int):
        """Initializes the RegressionEncoder.

        Parameters
        ----------
        D : int
            Dimensionality of the high-dimensional space (e.g., 10000).
        n_features : int
            Number of features in the input data.

        Returns
        -------
        RegressionEncoder
            The regression encoder object.
        """

        if D <= 0 or not isinstance(D, int):
            raise ValueError("Dimensionality D must be a positive integer.")

        if n_features <= 0 or not isinstance(n_features, int):
            raise ValueError("Number of features n_features must be a positive integer.")

        self.D = D
        self.n_features = n_features

        # Randomly chosen base hypervectors (B_k in paper)
        # These are generated from a normal distribution as suggested by the paper (B_kj ~ N(0,1)).
        # Shape: (n_features, D)
        self.base_hypervectors = np.random.randn(n_features, D)

        # Random biases b_i ~ U(0, 2pi)
        # Shape: (D,)
        self.biases = np.random.uniform(0, 2 * np.pi, D)

    def encode(self, feature_vector: np.ndarray) -> np.ndarray:
        """Encodes a single feature vector into a high-dimensional real-valued hypervector.

        Parameters
        ----------
        feature_vector : np.ndarray
            A 1D numpy array of shape (n_features,) representing the input features.

        Returns
        -------
        np.ndarray
            A 1D numpy array of shape (D,) representing the encoded high-dimensional real-valued hypervector.
        """

        # Ensure feature_vector is a 1D numpy array
        feature_vector = np.asarray(feature_vector).flatten()

        if feature_vector.shape[0] != self.n_features:
            raise ValueError(f"Input feature vector must have {self.n_features} features, but got {feature_vector.shape[0]}")

        # Compute dot products: F_bar . B_i for each B_i (column in base_hypervectors)
        # This results in a 1D array of shape (D,)
        dot_products = np.dot(feature_vector, self.base_hypervectors)

        # Apply the encoding formula: h_i = cos(dot_product + b_i) * sin(dot_product)
        encoded_vector = np.cos(dot_products + self.biases) * np.sin(dot_products)

        return encoded_vector


class RegressionModel:
    """Implements the RegHD multi-model regression algorithm.

    This class handles the training and prediction phases, including the management of cluster and regression models, 
    and optional quantization for efficiency.
    """
    def __init__(
        self, 
        D: int, 
        n_features: int, 
        k_models: int=8, 
        learning_rate: float=0.01, 
        iterations: int=20, 
        binary_threshold: float=0.0
    ):
        """Initializes the RegHDRegressor.

        Parameters
        ----------
        D : int
            Dimensionality of hypervectors.
        n_features : int
            Number of input features.
        k_models : int
            Number of cluster/regression models to use.
        learning_rate : float
            Alpha parameter for model updates.
        iterations : int
            Number of training iterations (epochs).
        binary_threshold : float
            Threshold for binarizing real-valued hypervectors (e.g., 0.0 for bipolar conversion).

        Returns
        -------
        RegressionModel
            The regression model object.
        """

        if D <= 0 or not isinstance(D, int):
            raise ValueError("Dimensionality D must be a positive integer.")

        if n_features <= 0 or not isinstance(n_features, int):
            raise ValueError("Number of features n_features must be a positive integer.")

        if k_models <= 0 or not isinstance(k_models, int):
            raise ValueError("Number of models k_models must be a positive integer.")

        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("Learning rate must be a positive number.")

        if iterations <= 0 or not isinstance(iterations, int):
            raise ValueError("Number of iterations must be a positive integer.")

        if not isinstance(binary_threshold, (int, float)):
            raise ValueError("Binary threshold must be a number.")

        self.D = D
        self.n_features = n_features
        self.k_models = k_models
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.binary_threshold = binary_threshold

        # Initialize the encoder for mapping input features to HD space
        self.encoder = RegressionEncoder(D, n_features)

        # Initialize cluster hypervectors (C_i) and regression model hypervectors (M_i)
        # As per the paper, we maintain both "integer" (float numpy array) and "binary" (Vector)
        # versions for cluster models.
        # Cluster models (binary for similarity search, float for updates)
        self.cluster_models_b = [Vector(vtype="bipolar", size=D) for _ in range(k_models)]
        self.cluster_models_int = [np.zeros(D, dtype=float) for _ in range(k_models)]

        # Initialize integer models from the binary ones to ensure consistency
        for i in range(self.k_models):
            self.cluster_models_int[i] = self.cluster_models_b[i].vector.astype(float)

        # Regression models (float for updates, binary for optional quantized prediction)
        self.regression_models_int = [np.zeros(D, dtype=float) for _ in range(k_models)]
        self.regression_models_b = [Vector(vtype="bipolar", vector=np.zeros(D)) for _ in range(k_models)]

        # Flag to control prediction quantization strategy
        self.quantized_prediction = False

    def _to_bipolar(self, vec_float: np.ndarray) -> Vector:
        """Converts a float numpy array to an hdlib Vector (bipolar). Values >= binary_threshold become 1, others become -1.
        """

        bipolar_array = np.where(vec_float >= self.binary_threshold, 1, -1)

        return Vector(vtype="bipolar", vector=bipolar_array)

    def _calculate_similarity(self, encoded_s_b: Vector, cluster_model_b: Vector) -> float:
        """Calculates similarity between a binary encoded input hypervector and a
        binary cluster model hypervector. For bipolar vectors, this is based on cosine similarity.
        """

        # Using cosine similarity as the primary metric for bipolar vectors in hdlib
        return 1.0 - encoded_s_b.dist(cluster_model_b, method="cosine")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Trains the RegHD regressor using the provided training data.

        Parameters
        ----------
        X : np.ndarray
            Input features, a 2D numpy array of shape (n_samples, n_features).
        y : np.ndarray
            Target values, a 1D numpy array of shape (n_samples,).
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.shape[1] != self.n_features:
            raise ValueError(f"Input X must have {self.n_features} features, but got {X.shape[1]}")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")

        print(f"Starting RegHD training with D={self.D}, k_models={self.k_models}, iterations={self.iterations}")

        for iteration in range(self.iterations):
            total_mse = 0.0

            for i in range(X.shape[0]):
                feature_vector = X[i]
                actual_output = y[i]

                # 1. Encode input data
                encoded_s_float = self.encoder.encode(feature_vector)
                encoded_s_b = self._to_bipolar(encoded_s_float)

                # 2. Check similarity with all binary cluster hypervectors
                similarities = np.array([self._calculate_similarity(encoded_s_b, cluster_b) for cluster_b in self.cluster_models_b])

                # 3. Normalize similarity values using softmax to get confidences
                confidences = softmax(similarities)

                # 4. Predict output value during training
                predicted_output = 0.0
                for j in range(self.k_models):
                    predicted_output += confidences[j] * np.dot(self.regression_models_int[j], encoded_s_float)

                # 5. Calculate prediction error
                error = actual_output - predicted_output
                total_mse += error**2

                # 6. Update Regression Models
                for j in range(self.k_models):
                    self.regression_models_int[j] += self.learning_rate * confidences[j] * error * encoded_s_float

                # 7. Update Cluster Models
                l_max_similarity = np.argmax(similarities)
                self.cluster_models_int[l_max_similarity] += (1 - confidences[l_max_similarity]) * encoded_s_float

            # After each epoch, quantize the updated models
            for j in range(self.k_models):
                self.cluster_models_b[j] = self._to_bipolar(self.cluster_models_int[j])

                if self.quantized_prediction:
                    self.regression_models_b[j] = self._to_bipolar(self.regression_models_int[j])

            avg_mse = total_mse / X.shape[0]

            print(f"Iteration {iteration+1}/{self.iterations}, Mean Squared Error (MSE): {avg_mse:.4f}")

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        """Predicts output values for new input data.

        Parameters
        ----------
        X_query : np.ndarray
            Input features, a 2D numpy array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted target values, a 1D numpy array of shape (n_samples,).
        """

        X_query = np.asarray(X_query, dtype=float)

        if X_query.shape[1] != self.n_features:
            raise ValueError(f"Input X_query must have {self.n_features} features, but got {X_query.shape[1]}")

        predictions = []

        for i in range(X_query.shape[0]):
            feature_vector = X_query[i]

            # 1. Encode input data
            encoded_s_float = self.encoder.encode(feature_vector)
            encoded_s_b = self._to_bipolar(encoded_s_float)

            # 2. Compute similarity with all binary cluster centers
            similarities = np.array([self._calculate_similarity(encoded_s_b, cluster_b) for cluster_b in self.cluster_models_b])

            # 3. Normalize similarity values
            confidences = softmax(similarities)

            # 4. Predict output value
            predicted_output = 0.0
            for j in range(self.k_models):
                if self.quantized_prediction:
                    predicted_output += confidences[j] * np.dot(self.regression_models_b[j].vector, encoded_s_b.vector)

                else:
                    predicted_output += confidences[j] * np.dot(self.regression_models_int[j], encoded_s_float)

            predictions.append(predicted_output)

        return np.array(predictions)

    def set_quantized_prediction_mode(self, enable: bool=True):
        """Sets whether prediction should use binarized regression models.
        """

        self.quantized_prediction = enable

        if enable:
            for j in range(self.k_models):
                self.regression_models_b[j] = self._to_bipolar(self.regression_models_int[j])
