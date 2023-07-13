"""Classification Model with Hyperdimensional Computing."""

import copy
import itertools
import multiprocessing as mp
import statistics
from functools import partial
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from hdlib import __version__
from hdlib.space import Space, Vector
from hdlib.arithmetic import bundle, permute
from hdlib.parser import kfolds_split


class Model(object):
    """Classification Model."""

    def __init__(
        self,
        size: int=10000,
        levels: int=2,
        vtype: str="bipolar",
    ) -> "Model":
        """Initialize a Model object.

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
            If the vector size is lower than 10,000 or the number of level vectors is lower than 2.

        Examples
        --------
        >>> from hdlib.model import Model
        >>> model = Model(size=10000, levels=100, vtype='bipolar')
        >>> type(model)
        <class 'hdlib.model.Model'>

        This creates a new Model object around a Space that can host random bipolar Vector objects with size 10,000.
        It also defines the number of level vectors to 100.

        Notes
        -----
        The classification model based on the hyperdimensional computing paradigm has been originally described in [1]_.

        .. [1] Cumbo, Fabio, Eleonora Cappelli, and Emanuel Weitschek. "A brain-inspired hyperdimensional computing approach 
        for classifying massive dna methylation data of cancer." Algorithms 13.9 (2020): 233.
        """

        if not isinstance(size, int):
            raise TypeError("Vectors size must be an integer number")

        if size < 10000:
            raise ValueError("Vectors size must be greater than or equal to 10000")

        # Register vectors dimensionality
        self.size = size

        if not isinstance(levels, int):
            raise TypeError("Levels must be an integer number")

        if levels < 2:
            raise ValueError("The number of levels must be greater than or equal to 2")

        # Register the number of levels
        self.levels = levels

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

    def __str__(self) -> None:
        """Print the Model object properties.

        Returns
        -------
        str
            A description of the Model object. It reports the vectors size, the vector type,
            the number of level vectors, the number of data points, and the number of class labels.

        Examples
        --------
        >>> from hdlib.model import Model
        >>> model = Model()
        >>> print(model)

                Class:   hdlib.model.Model
                Size:    10000
                Type:    bipolar
                Levels:  2
                Points:  0
                Classes:

                []

        Print the Model object properties. By default, the size of vectors in space is 10,000,
        their types is bipolar, and the number of level vectors is 2. The number of data points 
        and the number of class labels are empty here since no dataset has been processed yet.
        """

        return """
            Class:   hdlib.model.Model
            Version: {}
            Size:    {}
            Type:    {}
            Levels:  {}
            Points:  {}
            Classes:
            
            {}
        """.format(
            self.version,
            self.size,
            self.vtype,
            self.levels,
            len(self.space.memory()) - self.levels if self.space is not None else 0,
            np.array(list(self.classes))
        )

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
        """Initialize a new Model, then fit and cross-validate it. Used for size and levels hyperparameters tuning.

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
            Method used to compute the distance/similarity between vectors in space.
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
        model = Model(size=size, levels=levels, vtype=vtype)

        # Fit the model
        model.fit(points, labels)

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

        for y_indices, y_pred, _ in predictions:
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
    ) -> None:
        """Build a vector-symbolic architecture. Define level vectors and encode samples.

        Parameters
        ----------
        points : list
            List of lists with numerical data (floats).
        labels : list
            List with class labels. It has the same size of `points`.

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
        next_level = int((self.size / 2 / self.levels))
        change = int(self.size / 2)

        # Also define the interval level list
        level_list = list()

        # Get the minimum and maximum value in the input dataset
        min_value = np.inf
        max_value = np.NINF

        for point in points:
            min_point = min(point)
            max_point = max(point)

            if min_point < min_value:
                min_value = min_point

            if max_point > max_value:
                max_value = max_point

        gap = (max_value - min_value) / self.levels

        # Create level vectors
        for level_count in range(self.levels):
            level = "level_{}".format(level_count)

            if level_count == 0:
                base = np.full(self.size, -1)
                to_one = np.random.RandomState(seed=0).permutation(index_vector)[:change]

            else:
                to_one = np.random.RandomState(seed=0).permutation(index_vector)[:next_level]

            for index in to_one:
                base[index] = base[index] * -1

            vector = Vector(
                name=level,
                size=self.size,
                vtype=self.vtype,
                vector=copy.deepcopy(base)
            )

            self.space.insert(vector)

            right_bound = min_value + level_count * gap

            if level_count == 0:
                left_bound = right_bound

            else:
                left_bound = min_value + (level_count - 1) * gap

            level_list.append((left_bound, right_bound))

        # Encode all data points
        for point_position, point in enumerate(points):
            sum_vector = None

            for value_position, value in enumerate(point):

                if value == min_value:
                    level_count = 0

                elif value == max_value:
                    level_count = self.levels - 1

                else:
                    for level_position in range(len(level_list)):
                        left_bound, right_bound = level_list[level_position]

                        if left_bound <= value and right_bound > value:
                            level_count = level_position

                            break

                level_vector = self.space.get(names=["level_{}".format(level_count)])[0]

                roll_vector = permute(level_vector, rotate_by=value_position)

                if sum_vector is None:
                    sum_vector = roll_vector

                else:
                    sum_vector = bundle(sum_vector, roll_vector)

            # Add the hyperdimensional representation of the data point to the space
            sum_vector.name = "point_{}".format(point_position)
            self.space.insert(sum_vector)

            # Tag vector with its class label
            self.space.add_tag(name=sum_vector.name, tag=labels[point_position])

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
        """

        wrongly_predicted_training_vectors = list()

        wrong_predictions = list()

        for training_vector in training_vectors:
            # Vectors contain only their class info in tags
            true_class = list(training_vector.tags)[0]

            if true_class != None:
                closest_class = None
                closest_dist = np.NINF

                for class_vector in class_vectors:
                    # Compute the distance between the training points and the hyperdimensional representations of classes
                    with np.errstate(invalid="ignore", divide="ignore"):
                        distance = training_vector.dist(class_vector, method=distance_method)

                    if closest_class is None:
                        closest_class = list(class_vector.tags)[0]
                        closest_dist = distance

                    else:
                        if distance > closest_dist:
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
    ) -> Tuple[List[int], List[str], int]:
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
            same size of `test_indices` and the total number of retraining iterations used to retrain the classification model.

        Raises
        ------
        Exception
            If the number of test indices does not match the number of points retrieved from the space.
        """

        # List with test vector names
        test_points = list()

        # List with training vector names
        training_points = list()

        # Retrieve test and training vector names from vectors in the space
        for vector_name in self.space.memory():
            if vector_name.startswith("point_"):
                vector_id = int(vector_name.split("_")[-1])

                if vector_id in test_indices:
                    test_points.append(vector_name)

                else:
                    training_points.append(vector_name)

        if len(test_points) != len(test_indices):
            raise Exception("Unable to retrieve all the test vectors in space")

        class_vectors = list()

        for class_pos, class_label in enumerate(sorted(list(self.classes))):
            # Get training vectors for the current class label
            class_points = [vector for vector in self.space.get(names=training_points) if class_label in vector.tags]

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

        # Retrieve the test vectors from the space        
        test_vectors = self.space.get(names=test_points)

        # Make a copy of the vector representation of classes for retraining the model
        retraining_class_vectors = copy.deepcopy(class_vectors) if retrain > 0 else class_vectors

        # Count retraining iterations
        retraining_iterations = 0

        if retrain > 0:
            # Retrieve the training vectors from the space
            training_vectors = self.space.get(names=training_points)

            # Take track of the error rate while retraining the model
            model_error_rate, wrongly_predicted_training_vectors, wrong_predictions = self.error_rate(
                training_vectors,
                retraining_class_vectors,
                distance_method=distance_method
            )

            for _ in range(retrain):
                retraining_class_vectors_iter = copy.deepcopy(retraining_class_vectors)

                for training_vector in wrongly_predicted_training_vectors:
                    true_class = list(training_vector.tags)[0]

                    # Error mitigation
                    for vector_position, class_vector in enumerate(retraining_class_vectors_iter):
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

        for test_vector in sorted(test_vectors, key=lambda vector: test_indices.index(int(vector.name.split("_")[-1]))):
            closest_class = None
            closest_dist = np.NINF

            for class_vector in retraining_class_vectors:
                # Compute the distance between the test points and the hyperdimensional representations of classes
                with np.errstate(invalid="ignore", divide="ignore"):
                    distance = test_vector.dist(class_vector, method=distance_method)

                if closest_class is None:
                    closest_class = list(class_vector.tags)[0]
                    closest_dist = distance

                else:
                    if distance > closest_dist:
                        closest_class = list(class_vector.tags)[0]
                        closest_dist = distance

            prediction.append(closest_class)

        return test_indices, prediction, retraining_iterations

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
            Method used to compute the distance/similarity between vectors in space.
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
            - if the number of folds is a number < 1;
            - if the number of folds exceeds the number of data points.
        """

        if len(points) != len(labels):
            raise Exception("The number of data points does not match with the number of class labels")

        if len(set(labels)) < 2:
            raise Exception("The number of unique class labels must be > 1")

        if cv < 1:
            raise ValueError("The number of folds must be a positive number > 0")

        if cv > len(points):
            raise ValueError("The number of folds cannot exceed the number of data points")        

        split_indices = kfolds_split(len(points), cv)

        # Collect results from every self.predict call
        predictions = list()

        if n_jobs == 1:
            for test_indices in split_indices:
                _, test_predictions, retraining_iterations = self.predict(
                    test_indices,
                    distance_method=distance_method,
                    retrain=retrain
                )

                predictions.append((test_indices, test_predictions, retraining_iterations))

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
                        args=(test_indices,)
                    )
                    for test_indices in split_indices
                ]

                # Get results from jobs
                for job in jobs:
                    test_indices, test_predictions, retraining_iterations = job.get()

                    predictions.append((test_indices, test_predictions, retraining_iterations))

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
            Method used to compute the distance/similarity between vectors in space.
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
        """

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
            Method used to compute the distance/similarity between vectors in space.
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
    ) -> Tuple[Dict[str, int], Dict[int, float], int]:
        """Stepwise regression as backward variable elimination or forward variable selection.

        Parameters
        ----------
        points : list
            List of lists with numerical data (floats).
        features : list
            List of features
        labels : list
            List with class labels. It has the same size of `points`.
        method : {'backward', 'forward'}, default 'backward'
            Feature selection method.
        cv : int, default 5
            Number of folds for cross-validating the model.
        distance_method : {'cosine', 'euclidean', 'hamming'}, default 'cosine'
            Method used to compute the distance/similarity between vectors in space.
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
            A tuple with a dictionary with features and their importance in addition to the best score for each importance rank 
            and the best importance. In case of backward, the lower the better. In case of forward, the higher the better.
        """

        method = method.lower()

        if method not in ("backward", "forward"):
            raise ValueError("Stepwise method {} is not supported".format(method))

        # Initialize the importance of features to 0
        features_importance = {feature: {"importance": 0, "score": 0.0} for feature in features}

        features_indices = set(range(len(features)))

        # Take track of the last feature selection
        # Only in case of forward variable selection
        last_selection = set()

        prev_score = 0.0

        count_iter = 1

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

        return importances, scores, best_importance
