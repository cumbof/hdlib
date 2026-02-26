"""Classification with hdlib.

It implements the __hdlib.model.classification.ClassificationModel__ class object which allows to generate, fit, and test a classification model
built according to the Hyperdimensional Computing (HDC) paradigm as described in _Cumbo et al. 2020_ https://doi.org/10.3390/a13090233.

It also implements a stepwise regression model as backward and forward variable elimination techniques for selecting
relevant features in a dataset according to the same HDC paradigm.

The quantum version of this classification model is also provided here in __hdlib.model.classification.QuantumClassificationModel__
as described in _Cumbo et al. 2025_ https://doi.org/10.48550/arXiv.2511.12664."""

import copy
import itertools
import multiprocessing as mp
import os
import statistics
from math import log2
from functools import partial
from typing import Dict, List, Optional, Set, Tuple
from contextlib import nullcontext

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    Sampler,
    SamplerOptions,
    Session
)

from hdlib import __version__
from hdlib.space import Space
from hdlib.vector import Vector
from hdlib.arithmetic import bundle, permute

# Quantum functions
from hdlib.arithmetic.quantum import (
    encode as quantum_encode,
    bundle as quantum_bundle,
    permute as quantum_permute,
    run_hadamard_test,
    get_circuit_metrics,
)


class ClassificationModel(object):
    """Supervised Classification Model."""

    def __init__(
        self,
        size: int=10000,
        levels: int=2,
        vtype: str="bipolar",
    ) -> "ClassificationModel":
        """Initialize a ClassificationModel object.

        Parameters
        ----------
        size : int, default 10000
            The size of vectors used to create a Space and define Vector objects.
        levels : int, default 2
            The number of level vectors used to represent numerical data. It is 2 by default.
        vtype : {'binary', 'bipolar'}, default 'bipolar'
            The vector type in space, which is bipolar by default.

        Raises
        ------
        TypeError
            If the vector size or the number of levels are not integer numbers.
        ValueError
            If the number of level vectors is lower than 2.

        Examples
        --------
        >>> from hdlib.model import ClassificationModel
        >>> model = ClassificationModel(size=10000, levels=100, vtype='bipolar')
        >>> type(model)
        <class 'hdlib.model.ClassificationModel'>

        This creates a new ClassificationModel object around a Space that can host random bipolar Vector objects with size 10,000.
        It also defines the number of level vectors to 100.
        """

        if not isinstance(size, int):
            raise TypeError("Vectors size must be an integer number")

        # Register vectors dimensionality
        self.size = size

        if not isinstance(levels, int):
            raise TypeError("Levels must be an integer number")

        if levels < 2:
            raise ValueError("The number of levels must be greater than or equal to 2")

        # Register the number of levels
        self.levels = levels

        # Minimum and maximum values in the input dataset
        # This is used to define the level boundaries
        self.min_value = None
        self.max_value = None

        # List of level boundaries
        self.level_list = list()

        if vtype not in ("bipolar", "binary"):
            raise ValueError("Vectors type can be binary or bipolar only")

        # Register vectors type
        self.vtype = vtype.lower()

        # Hyperdimensional space
        self.space = None

        # Class labels
        self.classes = set()

        # Keep track of hdlib version
        self.version = __version__

    def __str__(self) -> str:
        """Print the ClassificationModel object properties.

        Returns
        -------
        str
            A description of the ClassificationModel object. It reports the vectors size, the vector type,
            the number of level vectors, the number of data points, and the number of class labels.

        Examples
        --------
        >>> from hdlib.model import ClassificationModel
        >>> model = ClassificationModel()
        >>> print(model)

                Class:   hdlib.model.classification.ClassificationModel
                Version: 0.1.17
                Size:    10000
                Type:    bipolar
                Levels:  2
                Points:  0
                Classes:

                []

        Print the ClassificationModel object properties. By default, the size of vectors in space is 10,000,
        their type is bipolar, and the number of level vectors is 2. The number of data points 
        and the number of class labels are empty here since no dataset has been processed yet.
        """

        return f"""
            Class:   hdlib.model.classification.ClassificationModel
            Version: {self.version}
            Size:    {self.size}
            Type:    {self.vtype}
            Levels:  {self.levels}
            Points:  {len(self.space.memory()) - self.levels if self.space is not None else 0}
            Classes:

            {np.array(list(self.classes))}
        """

    def _init_fit_predict(
        self,
        size: int=10000,
        levels: int=2,
        vtype: str="bipolar",
        points: Optional[List[List[float]]]=None,
        labels: Optional[List[str]]=None,
        cv: int=5,
        distance_method: str="cosine",
        retrain: int=0,
        n_jobs: int=1,
        metric: str="accuracy"
    ) -> Tuple[int, int, float]:
        """Initialize a new ClassificationModel, then fit and cross-validate it. Used for size and levels hyperparameters tuning.

        Parameters
        ----------
        size : int, default 10000
            The size of vectors used to create a Space and define Vector objects.
        levels : int, default 2
            The number of level vectors used to represent numerical data. It is 2 by default.
        vtype : {'binary', 'bipolar'}, default 'bipolar'
            The vector type in space, which is bipolar by default.
        points : list
            List of lists with numerical data (floats).
        labels : list
            List with class labels. It has the same size of `points`.
        cv : int, default 5
            Number of folds for cross-validating the model.
        distance_method : {'cosine', 'euclidean', 'hamming'}, default 'cosine'
            Method used to compute the distance between vectors in space.
        retrain : int, default 0
            Number of retraining iterations.
        n_jobs : int, default 1,
            Number of jobs for processing folds in parallel.
        metric: {'accuracy', 'f1', 'precision', 'recall'}, default 'accuracy'
            Metric used to evaluate the model.

        Returns
        -------
        tuple
            A tuple with the input size, the number of level vectors, and the model score
            according to the input metric.

        Raises
        ------
        ValueError
            If the provided metric is not supported.
        """

        # Available score metrics
        score_metrics = {
            "accuracy": accuracy_score,
            "f1": f1_score,
            "precision": precision_score,
            "recall": recall_score
        }

        metric = metric.lower()

        if metric not in score_metrics:
            raise ValueError("Score metric {} is not supported".format(metric))

        # Generate a new Model
        model = ClassificationModel(size=size, levels=levels, vtype=vtype)

        # Fit the model
        model.fit(points, labels=labels)

        # Cross-validate the model
        predictions = model.cross_val_predict(
            points,
            labels,
            cv=cv,
            distance_method=distance_method,
            retrain=retrain,
            n_jobs=n_jobs
        )

        # For each prediction, compute the score and return the average
        scores = list()

        for y_indices, y_pred, _, _, _, _ in predictions:
            y_true = [label for position, label in enumerate(labels) if position in y_indices]

            if metric == "accuracy":
                scores.append(score_metrics[metric](y_true, y_pred))

            else:
                # Use the average weighted to account for label imbalance
                scores.append(score_metrics[metric](y_true, y_pred, average="weighted"))

        return size, levels, statistics.mean(scores)

    def fit(
        self,
        points: List[List[float]],
        labels: List[str],
        seed: Optional[int]=None,
    ) -> None:
        """Build a vector-symbolic architecture. Define level vectors and encode samples.

        Parameters
        ----------
        points : list
            List of lists with numerical data (floats).
        labels : list
            List with class labels. It has the same size of `points`.
        seed : int, optional
            An optional seed for reproducibly generating the vectors numpy.ndarray randomly.

        Raises
        ------
        Exception
            - if there are not enough data points (the length of `points` is < 3);
            - if the length of `points` does not match the length of `labels`;
            - if there is only one class label.
        """

        if len(points) < 3:
            # This is based on the assumption that the minimum number of data points for training
            # the classification model is 2, while 1 data point is enough for the test set
            raise Exception("Not enough data points")

        if len(points) != len(labels):
            raise Exception("The number of data points does not match with the number of class labels")

        if len(set(labels)) < 2:
            raise Exception("The number of unique class labels must be > 1")

        self.classes = set(labels)

        # Initialize the hyperdimensional space so that it overwrites any existing space in Model
        self.space = Space(size=self.size, vtype=self.vtype)

        index_vector = range(self.size)

        change = int(self.size / 2)
        next_level = int((self.size / 2 / self.levels))

        # Also define the interval level list
        self.level_list = list()

        # Get the minimum and maximum value in the input dataset
        self.min_value = np.inf
        self.max_value = -np.inf

        for point in points:
            min_point = min(point)
            max_point = max(point)

            if min_point < self.min_value:
                self.min_value = min_point

            if max_point > self.max_value:
                self.max_value = max_point

        gap = (self.max_value - self.min_value) / self.levels

        if seed is None:
            rand = np.random.default_rng()

        else:
            # Conditions on random seed for reproducibility
            # numpy allows integers as random seeds
            if not isinstance(seed, int):
                raise TypeError("Seed must be an integer number")

            rand = np.random.default_rng(seed=seed)

        # Create level vectors
        for level_count in range(self.levels):
            level = "level_{}".format(level_count)

            if level_count == 0:
                base = np.full(self.size, -1 if self.vtype == "bipolar" else 0)
                to_one = rand.permutation(index_vector)[:change]

            else:
                to_one = rand.permutation(index_vector)[:next_level]

            for index in to_one:
                base[index] = base[index] * -1 if self.vtype == "bipolar" else base[index] + 1

            vector = Vector(
                name=level,
                size=self.size,
                vtype=self.vtype,
                vector=copy.deepcopy(base)
            )

            self.space.insert(vector)

            right_bound = self.min_value + level_count * gap

            if level_count == 0:
                left_bound = right_bound

            else:
                left_bound = self.min_value + (level_count - 1) * gap

            self.level_list.append((left_bound, right_bound))

        # Encode all data points
        for point_position, point in enumerate(points):
            point_vector = self._encode_point(point)

            # Add the hyperdimensional representation of the data point to the space
            point_vector.name = "point_{}".format(point_position)
            self.space.insert(point_vector)

            # Tag vector with its class label
            self.space.add_tag(name=point_vector.name, tag=labels[point_position])

    def _encode_point(self, point: List[float]) -> Vector:
        """Encode a single data point. It must be used after `fit()`.

        Parameters
        ----------
        point : list
            A data point.

        Returns
        -------
        Vector
            The encoded data point.
        """

        sum_vector = None

        for value_position, value in enumerate(point):
            level_count = 0

            if value == self.min_value:
                level_count = 0

            elif value == self.max_value:
                level_count = self.levels - 1

            else:
                for level_position in range(len(self.level_list)):
                    left_bound, right_bound = self.level_list[level_position]

                    if left_bound <= value and right_bound > value:
                        level_count = level_position

                        break

            level_vector = self.space.get(names=["level_{}".format(level_count)])[0]

            roll_vector = permute(level_vector, rotate_by=value_position)

            if sum_vector is None:
                sum_vector = roll_vector

            else:
                sum_vector = bundle(sum_vector, roll_vector)

        return sum_vector

    def error_rate(
        self,
        training_vectors: List[Vector],
        class_vectors: List[Vector],
        distance_method: str="cosine"
    ) -> Tuple[float, List[Vector], List[str]]:
        """Compute the error rate.

        Parameters
        ----------
        training_vectors : list
            List with Vector objects used for training the classification model.
        class_vectors : list
            List with the Vector representation of classes.
        distance_method : {'cosine', 'euclidean', 'hamming'}, default 'cosine'
            Method used to compute the distance between vectors in the space.

        Returns
        -------
        tuple
            A tuple with the error rate, the list of wrongly predicted Vector objects, and the list
            of wrong predictions with the same length of the list with wrongly predicted Vector objects.

        Raises
        ------
        ValueError
            - if the input `training_vectors` does not contain Vector objects;
            - if the input `class_vectors` does not contain Vector objects.
        Exception
            - if no training vectors have been provided;
            - if no class vectors have been provided.
        """

        if not training_vectors:
            raise Exception("No training vectors have been provided")

        if not class_vectors:
            raise Exception("No class vectors have been provided")

        wrongly_predicted_training_vectors = list()

        wrong_predictions = list()

        for class_vector in class_vectors:
            if not isinstance(class_vector, Vector):
                raise ValueError("The list of class vectors does not contain Vector objects")

        for training_vector in training_vectors:
            if not isinstance(training_vector, Vector):
                raise ValueError("The list of training vectors does not contain Vector objects")

            # Vectors contain only their class info in tags
            true_class = list(training_vector.tags)[0]

            if true_class != None:
                closest_class = None
                closest_dist = -np.inf

                for class_vector in class_vectors:
                    # Compute the distance between the training points and the hyperdimensional representations of classes
                    with np.errstate(invalid="ignore", divide="ignore"):
                        distance = training_vector.dist(class_vector, method=distance_method)

                    if closest_class is None:
                        closest_class = list(class_vector.tags)[0]
                        closest_dist = distance

                    else:
                        if distance < closest_dist:
                            closest_class = list(class_vector.tags)[0]
                            closest_dist = distance

                if closest_class != true_class:
                    wrongly_predicted_training_vectors.append(training_vector)

                    wrong_predictions.append(closest_class)

        model_error_rate = len(wrongly_predicted_training_vectors) / len(training_vectors)

        return model_error_rate, wrongly_predicted_training_vectors, wrong_predictions

    def predict(
        self,
        test_indices: List[int],
        distance_method: str="cosine",
        retrain: int=0
    ) -> Tuple[List[int], List[str], List[List[float]], int, float, List[Vector]]:
        """Supervised Learning. Predict the class labels of the data points in the test set.

        Parameters
        ----------
        test_indices : list
            Indices of data points in the list of points used with fit() to be used for testing the classification model.
            Note that all the other points will be used for training the model.
        distance_method : {'cosine', 'euclidean', 'hamming'}, default 'cosine'
            Method used to compute the distance between vectors in the space.
        retrain : int, default 0
            Maximum number of retraining iterations.

        Returns
        -------
        tuple
            A tuple with the input list `test_indices` in addition to a list with the predicted class labels with the 
            same size of `test_indices`, the distances between the test vectors and classes, the total number of 
            retraining iterations used to retrain the classification model, the model error rate, and the retrained 
            class vectors (i.e., the actual model).

        Raises
        ------
        ValueError
            If the number of retraining iterations is <0.
        Exception
            - if no test indices have been provided;
            - if no class labels have been provided while fitting the model;
            - if the number of test indices does not match the number of points retrieved from the space.

        Notes
        -----
        The supervised classification model based on the hyperdimensional computing paradigm has been originally described in [1]_.

        .. [1] Cumbo, Fabio, Eleonora Cappelli, and Emanuel Weitschek. "A brain-inspired hyperdimensional computing approach 
        for classifying massive dna methylation data of cancer." Algorithms 13.9 (2020): 233. 
        """

        if not test_indices:
            raise Exception("No test indices have been provided")

        if retrain < 0:
            raise ValueError("The number of retraining iterations must be >=0")

        if len(self.classes) == 0:
            raise Exception("No class labels found")

        # List with test vectors
        test_vectors = list()

        # List with training vectors
        training_vectors = list()

        # Retrieve test and training vectors from the space
        for vector_name in self.space.space:
            if vector_name.startswith("point_"):
                vector_id = int(vector_name.split("_")[-1])

                vector = self.space.space[vector_name]

                if vector_id in test_indices:
                    test_vectors.append(vector)

                else:
                    training_vectors.append(vector)

        if len(test_vectors) != len(test_indices):
            raise Exception("Unable to retrieve all the test vectors from the space")

        class_vectors = list()

        for class_pos, class_label in enumerate(self.classes):
            # Get training vectors for the current class label
            class_points = [vector for vector in training_vectors if class_label in vector.tags]

            # Build the vector representations of the current class
            class_vector = None

            for vector in class_points:
                if class_vector is None:
                    class_vector = vector

                else:
                    class_vector = bundle(class_vector, vector)

            class_vector.name = "class_{}".format(class_pos)

            class_vector.tags.add(class_label)

            class_vectors.append(class_vector)

        # Make a copy of the vector representation of classes for retraining the model
        retraining_class_vectors = copy.deepcopy(class_vectors) if retrain > 0 else class_vectors

        # Take track of the error rate in case of retraining the model
        model_error_rate, wrongly_predicted_training_vectors, wrong_predictions = self.error_rate(
            training_vectors,
            retraining_class_vectors,
            distance_method=distance_method
        )

        # Count retraining iterations
        retraining_iterations = 0

        if retrain > 0:
            for _ in range(retrain):
                retraining_class_vectors_iter = copy.deepcopy(retraining_class_vectors)

                for vector_position, training_vector in enumerate(wrongly_predicted_training_vectors):
                    true_class = list(training_vector.tags)[0]

                    # Error mitigation
                    for class_vector in retraining_class_vectors_iter:
                        if true_class in class_vector.tags:
                            class_vector.vector = class_vector.vector + training_vector.vector

                        elif wrong_predictions[vector_position] in class_vector.tags:
                            class_vector.vector = class_vector.vector - training_vector.vector

                retraining_error_rate, wrongly_predicted_training_vectors, wrong_predictions = self.error_rate(
                    training_vectors,
                    retraining_class_vectors_iter,
                    distance_method=distance_method
                )

                if model_error_rate < retraining_error_rate:
                    # Does not make sense to keep retraining if the error rate increases compared to the previous iteration
                    break

                # Take track of the error rate
                model_error_rate = retraining_error_rate

                # Use the retrained class vectors
                retraining_class_vectors = retraining_class_vectors_iter

                # Also take track of the number of retraining iterations
                retraining_iterations += 1

        prediction = list()
        distances = list()

        for test_vector in sorted(test_vectors, key=lambda vector: test_indices.index(int(vector.name.split("_")[-1]))):
            pred, dist = self._predict_vector(test_vector, retraining_class_vectors, distance_method=distance_method)

            prediction.append(pred)
            distances.append(dist)

        return test_indices, prediction, distances, retraining_iterations, model_error_rate, retraining_class_vectors

    def _predict_vector(
        self, 
        vector: Vector, 
        training_class_vectors: List[Vector], 
        distance_method: str="cosine"
    ) -> Tuple[str, List[float]]:
        """Predict the class of an input vector.

        Parameters
        ----------
        vector : Vector
            The input vector for prediction.
        training_class_vectors : list
            List of eventually retrained class vectors representing the classification model.

        Returns
        -------
        tuple
            The closest class as the prediction and a list of vector to classes distances.
        """

        closest_class = None
        closest_dist = -np.inf

        distances = list()

        for class_vector in training_class_vectors:
            # Compute the distance between the input vector and the hyperdimensional representations of classes
            with np.errstate(invalid="ignore", divide="ignore"):
                distance = vector.dist(class_vector, method=distance_method)

                distances.append(distance)

            if closest_class is None:
                closest_class = list(class_vector.tags)[0]
                closest_dist = distance

            else:
                if distance < closest_dist:
                    closest_class = list(class_vector.tags)[0]
                    closest_dist = distance

        return closest_class, distances

    def cross_val_predict(
        self,
        points: List[List[float]],
        labels: List[str],
        cv: int=5,
        distance_method: str="cosine",
        retrain: int=0,
        n_jobs: int=1
    ) -> List[Tuple[List[int], List[str], int]]:
        """Run `predict()` in cross validation.

        Parameters
        ----------
        points : list
            List of lists with numerical data (floats).
        labels : list
            List with class labels. It has the same size of `points`.
        cv : int, default 5
            Number of folds for cross-validating the model.
        distance_method : {'cosine', 'euclidean', 'hamming'}, default 'cosine'
            Method used to compute the distance between vectors in space.
        retrain : int, default 0
            Number of retraining iterations.
        n_jobs : int, default 1,
            Number of jobs for processing folds in parallel.

        Returns
        -------
        list
            A list with the results of `predict()` for each fold.

        Raises
        ------
        Exception
            - if the number of data points does not match with the number of class labels;
            - if there is only one class label.
        ValueError
            - if the number of folds is a number < 2;
            - if the number of folds exceeds the number of data points;
            - if the number of retraining iterations is <0.
        """

        if len(points) != len(labels):
            raise Exception("The number of data points does not match with the number of class labels")

        if len(set(labels)) < 2:
            raise Exception("The number of unique class labels must be > 1")

        if cv < 2:
            raise ValueError("Not enough folds for cross-validating the model. Please use a minimum of 2 folds")

        if cv > len(points):
            raise ValueError("The number of folds cannot exceed the number of data points")

        if retrain < 0:
            raise ValueError("The number of retraining iterations must be >=0")

        # Use all the available resources if n_job < 1
        n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs

        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)

        # Collect results from every self.predict call
        predictions = list()

        if n_jobs == 1:
            for _, test_indices in kf.split(points, labels):
                test_indices = test_indices.tolist()

                _, test_predictions, test_distances, retraining_iterations, model_error_rate, training_class_vectors = self.predict(
                    test_indices,
                    distance_method=distance_method,
                    retrain=retrain
                )

                predictions.append((test_indices, test_predictions, test_distances, retraining_iterations, model_error_rate, training_class_vectors))

        else:
            predict_partial = partial(
                self.predict,
                distance_method=distance_method,
                retrain=retrain
            )

            # Run prediction on folds in parallel
            with mp.Pool(processes=n_jobs) as pool:
                jobs = [
                    pool.apply_async(
                        predict_partial,
                        args=(test_indices.tolist(),)
                    )
                    for _, test_indices in kf.split(points, labels)
                ]

                # Get results from jobs
                for job in jobs:
                    test_indices, test_predictions, test_distances, retraining_iterations, model_error_rate, training_class_vectors = job.get()

                    predictions.append((test_indices, test_predictions, test_distances, retraining_iterations, model_error_rate, training_class_vectors))

        return predictions

    def auto_tune(
        self,
        points: List[List[float]],
        labels: List[str],
        size_range: range,
        levels_range: range,
        cv: int=5,
        distance_method: str="cosine",
        retrain: int=0,
        n_jobs: int=1,
        metric: str="accuracy"
    ) -> Tuple[int, int, float]:
        """Automated hyperparameters tuning. Perform a Parameter Sweep Analysis (PSA) on space dimensionality and number of levels.

        Parameters
        ----------
        points : list
            List of lists with numerical data (floats).
        labels : list
            List with class labels. It has the same size of `points`.
        size_range : range
            Range of dimensionalities for performing PSA.
        levels_range : range
            Range of number of levels for performing PSA.
        cv : int, default 5
            Number of folds for cross-validating the model.
        distance_method : {'cosine', 'euclidean', 'hamming'}, default 'cosine'
            Method used to compute the distance between vectors in space.
        retrain : int, default 0
            Number of retraining iterations.
        n_jobs : int, default 1,
            Number of jobs for processing folds in parallel.
        metric: {'accuracy', 'f1', 'precision', 'recall'}, default 'accuracy'
            Metric used to evaluate the model.

        Returns
        -------
        tuple
            A tuple with the best size and levels according to the accuracies of the cross-validated models.

        Raises
        ------
        ValueError
            - if the number of class labels does not match with the number of data points;
            - if the number of specified folds for cross-validating the model is lower than 2;
            - if the number of folds exceeds the number of data points;
            - if the number of retraining iterations is <0.
        Exception
            - if no data points have been provided in input;
            - if no class labels have been provided in input.
        """

        if not points:
            raise Exception("No data points have been provided")

        if not labels:
            raise Exception("No class labels have been provided")

        if len(points) != len(labels):
            raise ValueError("The number of class labels must match with the number of data points")

        if cv < 2:
            raise ValueError("Not enough folds for cross-validating the model. Please use a minimum of 2 folds")

        if cv > len(points):
            raise ValueError("The number of folds cannot exceed the number of data points")

        if retrain < 0:
            raise ValueError("The number of retraining iterations must be >=0")

        # Use all the available resources if n_job < 1
        n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs

        partial_init_fit_predict = partial(
            self._init_fit_predict,
            vtype=self.vtype,
            points=points,
            labels=labels,
            cv=cv,
            distance_method=distance_method,
            retrain=retrain,
            n_jobs=1,
            metric=metric
        )

        best_metric = None
        best_size = None
        best_levels = None

        with mp.Pool(processes=n_jobs) as pool:
            jobs = [
                pool.apply_async(
                    partial_init_fit_predict,
                    args=(size, levels,)
                )
                for size, levels in list(itertools.product(size_range, levels_range)) \
                    if size > len(points) and levels > 1
            ]

            # Get results from jobs
            for job in jobs:
                job_size, job_levels, job_metric = job.get()

                if best_metric is None:
                    best_metric = job_metric
                    best_size = job_size
                    best_levels = job_levels

                else:
                    if job_metric > best_metric:
                        # Get the size and levels of the classification model with the best score metric
                        best_metric = job_metric
                        best_size = job_size
                        best_levels = job_levels

                    elif job_metric == best_metric:
                        # Minimize the number of levels in this case
                        if job_levels < best_levels:
                            best_size = job_size
                            best_levels = job_levels

                        elif job_levels == best_levels:
                            # Minimize the size in this case
                            if job_size < best_size:
                                best_size = job_size

        return best_size, best_levels, best_metric

    def _stepwise_regression_iter(
        self,
        features_indices: Set[int],
        points: List[List[float]],
        labels: List[str],
        cv: int=5,
        distance_method: str="cosine",
        retrain: int=0,
        metric: str="accuracy"
   ) -> Tuple[Set[float], float]:
        """Just a single iteration of the feature selection method.

        Parameters
        ----------
        features_indices : set
            Indices of features for shaping points.
        points : list
            List of lists with numerical data (floats).
        labels : list
            List with class labels. It has the same size of `points`.
        cv : int, default 5
            Number of folds for cross-validating the model.
        distance_method : {'cosine', 'euclidean', 'hamming'}, default 'cosine'
            Method used to compute the distance between vectors in space.
        retrain : int, default 0
            Number of retraining iterations.
        metric: {'accuracy', 'f1', 'precision', 'recall'}, default 'accuracy'
            Metric used to evaluate the model.

        Returns
        -------
        tuple
            A tuple with the considered features and the score of the classification model based on the provided metric.
        """

        data_points = [[point[i] for i in range(len(point)) if i in features_indices] for point in points]

        _, _, score = self._init_fit_predict(
            size=self.size,
            levels=self.levels,
            vtype=self.vtype,
            points=data_points,
            labels=labels,
            cv=cv,
            distance_method=distance_method,
            retrain=retrain,
            n_jobs=1,
            metric=metric
        )

        return features_indices, score

    def stepwise_regression(
        self,
        points: List[List[float]],
        features: List[str],
        labels: List[str],
        method: str="backward",
        cv: int=5,
        distance_method: str="cosine",
        retrain: int=0,
        n_jobs: int=1,
        metric: str="accuracy",
        threshold: float=0.6,
        uncertainty: float=5.0,
        stop_if_worse: bool=False
    ) -> Tuple[Dict[str, int], Dict[int, float], int, int]:
        """Stepwise regression as backward variable elimination or forward variable selection.

        Parameters
        ----------
        points : list
            List of lists with numerical data (floats).
        features : list
            List of features.
        labels : list
            List with class labels. It has the same size of `points`.
        method : {'backward', 'forward'}, default 'backward'
            Feature selection method.
        cv : int, default 5
            Number of folds for cross-validating the model.
        distance_method : {'cosine', 'euclidean', 'hamming'}, default 'cosine'
            Method used to compute the distance between vectors in space.
        retrain : int, default 0
            Number of retraining iterations.
        n_jobs : int, default 1,
            Number of jobs for processing models in parallel.
        metric: {'accuracy', 'f1', 'precision', 'recall'}, default 'accuracy'
            Metric used to evaluate the model.
        threshold : float, default 0.6
            Threshold on the model score metric. Stop running the feature selection if the best 
            reached score is lower than this threshold.
        uncertainty : float, default 5.0
            Uncertainty percentage threshold for comparing models metrics.
        stop_if_worse : bool, default False
            Stop running the feature selection if the accuracy reached at the iteration i is lower than the accuracy reached at i-1.

        Returns
        -------
        tuple
            A tuple with a dictionary with features and their importance in addition to the best score for each importance rank, 
            the best importance, and the total mount of ML models built and evaluated. For what concerns the importance, in case of 
            `method='backward'`, the lower the better. In case of `method='forward'`, the higher the better.

        Raises
        ------
        ValueError
            - if there are not enough features for running the feature selection;
            - if the number of class labels does not match with the number of data points;
            - if the specified feature selection method is not supported;
            - if the number of specified folds for cross-validating the model is lower than 2;
            - if the number of folds exceeds the number of data points;
            - if the number of retraining iterations is <0;
            - if the threshold is negative or greater than 1.0;
            - if the uncertainty percentage is negative or greater than 100.0.
        Exception
            - if no data points have been provided in input;
            - if no class labels have been provided in input.
        """

        if not points:
            raise Exception("No data points have been provided")

        if len(features) < 2:
            raise ValueError("Not enough features for running a feature selection")

        if not labels:
            raise Exception("No class labels have been provided")

        if len(points) != len(labels):
            raise ValueError("The number of class labels must match with the number of data points")

        method = method.lower()

        if method not in ("backward", "forward"):
            raise ValueError("Stepwise method {} is not supported".format(method))

        if cv < 2:
            raise ValueError("Not enough folds for cross-validating the model. Please use a minimum of 2 folds")

        if cv > len(points):
            raise ValueError("The number of folds cannot exceed the number of data points")

        if retrain < 0:
            raise ValueError("The number of retraining iterations must be >=0")

        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Invalid threshold! It must be >= 0.0 and <= 1.0")

        if uncertainty < 0.0 or uncertainty > 100.0:
            raise ValueError("Invalid uncertainty percentage! It must be >= 0.0 and <= 100.0")

        # Use all the available resources if n_job < 1
        n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs

        # Initialize the importance of features to 0
        features_importance = {feature: {"importance": 0, "score": 0.0} for feature in features}

        features_indices = set(range(len(features)))

        # Take track of the last feature selection
        # Only in case of forward variable selection
        last_selection = set()

        prev_score = 0.0

        count_iter = 1

        # Count the total amount of ML models built and evaluated
        count_models = 0

        while features_indices:
            if method == "backward":
                features_set_size = len(features_indices) - 1

                if len(features_indices) == 1:
                    break

            elif method == "forward":
                features_set_size = count_iter

                if features_set_size >= len(features_indices):
                    break

            if features_set_size > 0:
                best_score = 0.0
                classification_results = list()

                partial_stepwise_regression_iter = partial(
                    self._stepwise_regression_iter,
                    points=points,
                    labels=labels,
                    cv=cv,
                    distance_method=distance_method,
                    retrain=retrain,
                    metric=metric
                )

                with mp.Pool(processes=n_jobs) as pool:
                    # Get all combinations of features of a given size
                    jobs = [
                        pool.apply_async(
                            partial_stepwise_regression_iter,
                            args=(features_set,)
                        )
                        for features_set in itertools.combinations(features_indices, features_set_size)
                    ]

                    # Get results from jobs
                    for job in jobs:
                        job_features_set, job_score = job.get()

                        if job_score >= best_score:
                            # Keep track of the best score
                            best_score = job_score

                        classification_results.append((job_features_set, job_score))

                        count_models += 1

                selection = set()

                best_scores = list()

                for features_set, score in classification_results:
                    if score >= best_score - (best_score * uncertainty / 100.0):
                        # Keep track of the missing features in models that reached the best score
                        if method == "backward":
                            selection.update(features_indices.difference(features_set))

                        elif method == "forward":
                            selection.update(features_set)

                        best_scores.append(score)

                if method == "forward" and last_selection:
                    if len(last_selection) == len(selection) and len(last_selection.difference(selection)) == 0:
                        break

                last_selection = selection

                avg_score = statistics.mean(best_scores)

                if method == "backward":
                    # Keep decreasing the importance of worst features detected in previous iterations
                    for feature in features_importance:
                        if features_importance[feature]["importance"] >= 1:
                            features_importance[feature]["importance"] += 1

                # Set the importance of the selected features
                for feature_index in selection:
                    features_importance[features[feature_index]]["importance"] += 1
                    features_importance[features[feature_index]]["score"] = avg_score

                if method == "backward":
                    features_indices = features_indices.difference(selection)

                elif method == "forward":
                    features_indices = selection

                if stop_if_worse:
                    if best_score < prev_score - (prev_score * uncertainty / 100.0):
                        prev_score = avg_score

                        break

                prev_score = avg_score

                count_iter += 1

                if best_score < threshold:
                    break

        importances = dict()

        scores = dict()

        for feature in features_importance:
            importances[feature] = features_importance[feature]["importance"]

            scores[features_importance[feature]["importance"]] = features_importance[feature]["score"]

        best_importance = sorted(scores.keys(), key=lambda imp: scores[imp])[-1]

        return importances, scores, best_importance, count_models


class QuantumClassificationModel(object):
    """Supervised Quantum Classification Model."""

    def __init__(
        self,
        size: int=64,
        levels: int=2,
        seed: int=42,
        shots: int=1024,
        channel: Optional[str]=None,
        instance: Optional[str]=None,
        backend: Optional[str]=None,
        api_key: Optional[str]=None,
        noise_model_from: Optional[str]=None
    ) -> "QuantumClassificationModel":
        """Initialize a QuantumClassificationModel object.
        Run the classification model on a simulator with Qiskit by default.
        It can interact with specific IBM channels, instances, and backends if specified (it requires an IBM account).

        Parameters
        ----------
        size : int, default 64
            The vectors dimensionality as power of 2.
        levels : int, default 2
            Number of level vectors.
        seed : int, default 42
            Seed for reproducibility.
        shots : int, default 1024
            The number of times to run the quantum circuit for the Hadamard test.
        channel : str, default None, optional
            IBM channel.
        instance : str, default None, optional
            IBM instance. Required in case of specific channel only.
        backend : str, default None, optional
            IBM backend (e.g., "ibm_cleveland"). Required in case of specific instance only.
            If `instance` is not None, this is "least_busy" by default.
        api_key : str, default None, optional
            IBM API key. Required in case of specific backend only.
        noise_model_from : str, default None, optional
            The name of a real IBM backend (e.g., "ibm_cleveland") to build a noise model from the simulation.
            If provided, `api_key` is required. This parameter is ignored if `channel`, `instance`, and `backend` are provided for hardware execution.
            Noise models are retrieved from the "ibm_quantum_platform" channel.

        Raises
        ------
        ValueError
            If the vector dimensionality `size` is not a power of 2.
        TypeError
            If seed is not an integer.

        Examples
        --------
        >>> from hdlib.model import QuantumClassificationModel
        >>> model = QuantumClassificationModel(size=32, levels=2, oaa_rounds=2)
        >>> type(model)
        <class 'hdlib.model.QuantumClassificationModel'>

        This creates a new QuantumClassificationModel object with random bipolar vectors with size 32 and 2 level vectors.
        """

        if not ((size > 0) and ((size & (size - 1)) == 0)):
            # Check if a the vector dimensionality is a power of 2.
            raise ValueError("The vector dimensionality must be a power of 2.")

        self.size = size
        self.levels = levels
        self.shots = shots

        # Vectors must be bipolar here
        self.vtype = "bipolar"

        # Keep track of the level vectors
        self.level_hvs = list()

        # Keep track of class prototype vectors
        # This is filled up during `fit`
        self.prototypes = list()

        if channel is None:
            # Use a simulator if no channel is specified
            noise_model = None

            if noise_model_from:
                if not api_key:
                    raise ValueError("`api_key` must be provided to fetch backend properties for a noise model.")

                # Initialize a temporary service connection
                # Always use "ibm_quantum_platform" to fetch the backend properties
                noise_service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)

                # Retrieve a backend
                # We only need its noise model
                backend_for_noise = noise_service.backend(noise_model_from)

                # Finally, define the noise model
                noise_model = NoiseModel.from_backend(backend_for_noise)

            # Use a simulator if no channel is specified
            # This can be noise-free or use a specific noise model
            self.backend = AerSimulator(noise_model=noise_model)

        else:
            # Initialize a quantum runtime service for a specific IBM QC channel, instance, and backend
            service = QiskitRuntimeService(channel=channel, token=api_key, instance=instance)

            # The backend is the "least_busy" by default
            if backend is None:
                self.backend = service.least_busy(operational=True, simulator=False)

            else:
                self.backend = service.backend(backend)

        # Conditions on random seed for reproducibility
        # numpy allows integers as random seeds
        if not isinstance(seed, int):
            raise TypeError("Seed must be an integer number")

        self.seed = seed

        # Keep track of hdlib version
        self.version = __version__

    def __str__(self) -> str:
        """Print the QuantumClassificationModel object properties.

        Returns
        -------
        str
            A description of the QuantumClassificationModel object. It reports the vectors size, the vector type,
            the number of level vectors, the number of shots, and the number of OAA rounds.

        Examples
        --------
        >>> from hdlib.model import QuantumClassificationModel
        >>> model = QuantumClassificationModel()
        >>> print(model)

                Class:      hdlib.model.classification.QuantumClassificationModel
                Version:    2.0.0
                Size:       64
                Type:       bipolar
                Levels:     2
                Shots:      1024
                OAA Rounds: 1

        Print the QuantumClassificationModel object properties.
        """

        return f"""
            Class:      hdlib.model.classification.QuantumClassificationModel
            Version:    {self.version}
            Size:       {self.size}
            Type:       {self.vtype}
            Levels:     {self.levels}
            Shots:      {self.shots}
            OAA Rounds: {self.oaa_rounds}
        """

    def _build_quantum_sample_encoder(self, sample_row: List[float], level_vectors: List[np.ndarray], D: int) -> List[QuantumCircuit]:
        """Creates a single quantum circuit that encodes one real-valued sample by quantumly permuting its feature vectors.

        Parameters
        ----------
        sample_row : list
            Single sample as list of numerical values (float).
        level_vectors : list
            List of level vector.
        D : int
            Vector dimensionality.

        Returns
        -------
        List[QuantumCircuit]
            The list of quantum feature circuits for encoding the input sample.
        """

        num_qubits = int(log2(D))
        feature_circuits = list()

        for i, value in enumerate(sample_row):
            level_index = int(value * (len(level_vectors) - 1))
            level_vec = level_vectors[level_index]

            # Create a quantum circuit for this single feature
            feature_qc = quantum_encode(level_vec, label=f"Feature_{i}")

            # Apply the permutation based in the feature index
            feature_qc = quantum_permute(feature_qc, num_qubits, shift=i)

            # Report circuit metrics
            #print(f"Positional permutation metrics: {get_circuit_metrics(feature_qc, num_qubits, self.backend, optimization_level=3)}")

            feature_circuits.append(feature_qc)

        # The feature circuits are bundled all together with the feature circuits of the other samples
        # belonging to the same class to build the circuit representation of that class.
        return feature_circuits

    def fit(
        self,
        train_points: List[List[float]],
        train_labels: List[str]
    ) -> None:
        """Build a vector-symbolic architecture. Define level vectors, encode samples, and build prototypes.

        Parameters
        ----------
        train_points : list
            List of lists with numerical data points (floats).
        train_labels : list
            List with class labels. It has the same size of `points`.
        """

        def _generate_level_vectors(D, num_levels, rng):
            """Generates a set of bipolar vectors for discretizing real numbers.
            """

            level_vectors = [rng.choice([-1, 1], size=D)]

            change = int(D / 2)
            next_level = int((D / 2 / num_levels))

            for i in range(1, num_levels):
                prev_vec = level_vectors[i-1].copy()

                if i-1 == 0:
                    flip_indices = rng.choice(D, size=change, replace=False)

                else:
                    flip_indices = rng.choice(D, size=next_level, replace=False)

                prev_vec[flip_indices] *= -1
                level_vectors.append(prev_vec)

            return level_vectors

        rand = np.random.default_rng(seed=self.seed)

        # Create level vectors
        self.level_hvs = _generate_level_vectors(self.size, self.levels, rand)

        self.classes_ = sorted(list(set(train_labels)))
        self.prototypes = list()

        for c in self.classes_:
            # Building quantum prototype for Class `c`
            class_samples = [sample for pos, sample in enumerate(train_points) if train_labels[pos] == c]

            # Building quantum samples' feature encoder circuits
            sample_encoders = [encoder for sample in class_samples for encoder in self._build_quantum_sample_encoder(sample, self.level_hvs, self.size)]

            # Bundling
            prototype_circuit = quantum_bundle(sample_encoders, method="average")
            prototype_circuit.name = f"Prototype_{c}"

            # Report circuit metrics
            #print(f"Class prototype bundling metrics: {get_circuit_metrics(prototype_circuit, int(log2(self.size)), self.backend, optimization_level=3)}")

            # The prototype is the circuit that prepares the state
            self.prototypes.append(prototype_circuit)

    def predict(self, test_points: List[List[float]]) -> Tuple[List[str], List[List[float]]]:
        """Predict the class labels of the data points in the test set.

        Parameters
        ----------
        test_points : list
            Test data points.

        Returns
        -------
        Tuple
            A list with the predicted class labels in the same order of data points in `test_points`,
            and a list of similarities between the test samples and the class prototypes.

        Raises
        ------
        RuntimeError
            If `predict` is called before `fit`.
        """

        if not hasattr(self, "classes_"):
            raise RuntimeError("You must call fit before calling predict.")

        # Check whether the test should be performed on a simulator or on the quantum hardware
        is_simulated = isinstance(self.backend, AerSimulator)

        predictions = list()
        similarities = list()

        # For hardware, manage a single session for all prediction tasks
        with Session(backend=self.backend) if not is_simulated else nullcontext() as session:
            sampler = None

            # Use the session-based Sampler if on hardware
            if session:
                options = SamplerOptions()

                # Set the default number of shots
                options.default_shots = self.shots

                # Enable dynamical decoupling
                options.dynamical_decoupling.enable = True
                options.dynamical_decoupling.sequence_type = "XpXm"

                # Enable gate twirling
                options.twirling.enable_gates = True

                sampler = Sampler(mode=session, options=options)

            for sample in test_points:
                # Get the list of feature circuits for the test sample
                sample_features_circuits = self._build_quantum_sample_encoder(sample, self.level_hvs, self.size)

                # Bundle them into a single query circuit
                query_circuit = quantum_bundle(sample_features_circuits, method="average")

                # Report circuit metrics
                #print(f"Query bundling metrics: {get_circuit_metrics(query_circuit, int(log2(self.size)), self.backend, optimization_level=3)}")

                sample_similarities = list()

                # Run the Hadamard test between the query circuit and the class prototypes
                for prototype in self.prototypes:
                    # Prototype is a QuantumCircuit
                    similarity, _ = run_hadamard_test(query_circuit, prototype, self.backend, shots=self.shots, seed=self.seed, sampler=sampler)

                    sample_similarities.append(similarity)

                predictions.append(self.classes_[int(np.argmax(sample_similarities))])
                similarities.append(sample_similarities)

        return predictions, similarities
