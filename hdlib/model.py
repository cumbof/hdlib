"""
Classification model
"""

import copy
import multiprocessing as mp
from functools import partial
from typing import List, Optional, Tuple

import numpy as np

from hdlib import __version__
from hdlib.space import Space, Vector
from hdlib.arithmetic import bundle, permute
from hdlib.parser import kfolds_split


class Model(object):
    """
    Classification Model
    """

    def __init__(
        self,
        size: int=10000,
        levels: Optional[int]=None,
        vtype: str="bipolar",
        seed: Optional[int]=None,
    ) -> "Model":
        """
        Initialize a Model object

        :param size:        Vector size or dimensionality
        :param levels:      Number of levels. Try to automatically establish a good number of levels if None
        :param vtype:       Vector type: bipolar or binary
        :param seed:        Random seed for reproducibility
        :return:            A Model object
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

        # Register random seed for reproducibility
        self.seed = seed

        # Hyperdimensional space
        self.space = None

        # Class labels
        self.classes = set()

        # Keep track of hdlib version
        self.version = __version__

    def __str__(self) -> None:
        """
        Print the Model object properties
        """

        return """
            Class:   hdlib.model.Model
            Version: {}
            Seed:    {}
            Size:    {}
            Type:    {}
            Levels:  {}
            Points:  {}
            Classes:
            
            {}
        """.format(
            self.version,
            self.seed,
            self.size,
            self.vtype,
            self.levels,
            len(self.space.memory()) - self.levels,
            np.array(list(self.classes))
        )

    def fit(
        self,
        points: List[List[float]],
        labels: List[str],
    ) -> None:
        """
        Build a vector-symbolic architecture
        Define level vectors and encode samples

        :param points:  List of data points
        :param labels:  Class labels
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

    def predict(
        self,
        test_indices: List[int],
        exclude_points: Optional[List[int]]=None,
        distance_method: str="cosine",
        retrain: int=0
    ) -> Tuple[List[int], List[str]]:
        """
        Predict the class labels of the data points in the test set

        :param test_indices:    Indices of data points in the list of points used with fit() to be used for testing the classification model.
                                Note that all the other points will be used for training the model except those specified in exclude_points
        :param exclude_points:  Optional list of indices of data points to exclude from the training set
        :param distance_method: Method used to compute the distance between vectors in the space
                                Look at the dist() method of htlib.space.Vector class for a list of supported distance methods
        :param retrain:         Maximum number of retraining iterations
        :return:                The list test_indices and a list with the predicted class labels with the same size of test_indices
        """

        # List with test vector names
        test_points = list()

        # List with training vector names
        training_points = list()

        if exclude_points is None:
            exclude_points = list()

        # Retrieve test and training vector names from vectors in the space
        for vector_name in self.space.memory():
            if vector_name.startswith("point_"):
                vector_id = int(vector_name.split("_")[-1])

                if vector_id in test_indices:
                    test_points.append(vector_name)

                else:
                    if vector_id not in exclude_points:
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

        retraining_class_vectors = copy.deepcopy(class_vectors)

        # Retrain model
        retraining_iterations = 0

        # Take track of the predictions in the last retraining iteration 
        last_predictions = dict()

        while retrain + 1 > 0:
            prediction = list()

            if retraining_iterations > 0:
                for test_point in last_predictions:
                    true_class = None

                    # Retrieve the test vector tags
                    for test_vector in test_vectors:
                        if test_vector.name == test_point:
                            # Vectors contain only their class info in tags
                            true_class = list(test_vector.tags)[0]

                            break

                    if true_class != None:
                        # In case the test point has been wrongly predicted
                        if last_predictions[test_point] != true_class:
                            for class_vector in retraining_class_vectors:
                                if last_predictions[test_point] in class_vector.tags:
                                    class_vector.vector = class_vector.vector - test_vector.vector

                                if true_class in class_vector.tags:
                                    class_vector.vector = class_vector.vector + test_vector.vector

            for test_vector in sorted(test_vectors, key=lambda vector: test_indices.index(int(vector.name.split("_")[-1]))):
                closest_class = None
                closest_dist = np.NINF

                for class_vector in retraining_class_vectors:
                    # Compute the distance between the test points and 
                    # the hyperdimensional representations of classes
                    distance = test_vector.dist(class_vector, method=distance_method)

                    if closest_class is None:
                        closest_class = list(class_vector.tags)[0]
                        closest_dist = distance

                    else:
                        if distance > closest_dist:
                            closest_class = list(class_vector.tags)[0]
                            closest_dist = distance

                last_predictions[test_vector.name] = closest_class

                prediction.append(closest_class)

            retraining_iterations += 1

        return test_indices, prediction

    def cross_val_predict(
        self,
        points: List[List[float]],
        labels: List[str],
        exclude_points: Optional[List[int]]=None,
        cv: int=5,
        distance_method: str="cosine",
        retrain: int=0,
        n_jobs: int=1
    ) -> List[List[str]]:
        """
        Run predict() in cross validation

        :param points:          List with data points. Same used for fit()
        :param labels:          Class labels. Same used for fit()
        :param exclude_points:  Optional list of indices of data points to exclude from the training set
        :param cv:              Number of folds for the cross validation
        :param distance_method: Method used to compute the distance between vectors in the space
                                Look at the dist() method of htlib.space.Vector class for a list of supported distance methods
        :param retrain:         Maximum number of retraining iterations
        :param n_jobs:          Number of jobs for processing folds in parallel
        :return:                A list with the result of predict() for each fold
        """

        # TODO Excluded points must be removed before comparing the number of folds with the number of points
        #      and before running kfolds_split(). Otherwise, it would potentially reduce the training and test sets
        #      so that it does not make sense to run self.predict() anymore

        if len(points) != len(labels):
            raise Exception("The number of data points does not match with the number of class labels")

        if len(set(labels)) < 2:
            raise Exception("The number of unique class labels must be > 1")

        if cv < 1:
            raise ValueError("The number of folds must be a positive number > 0")

        if cv > len(points):
            raise ValueError("The number of folds cannot exceed the number of data points")        

        split_indices = kfolds_split(len(points), cv)

        predict_partial = partial(
            self.predict,
            exclude_points=exclude_points,
            distance_method=distance_method,
            retrain=retrain
        )

        # Collect results from every self.predict call
        predictions = list()

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
                test_indices, test_predictions = job.get()

                predictions.append((test_indices, test_predictions))

        return predictions
