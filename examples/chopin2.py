#!/usr/bin/env python3
"""
Implementation of chopin2 ML model with hdlib
https://github.com/cumbof/chopin2
"""

__author__ = ("Fabio Cumbo (fabio.cumbo@gmail.com)")

__version__ = "0.1.0"
__date__ = "Jun 1, 2023"

import argparse as ap
import copy
import errno
import os
import sys
import tempfile
import time
import unittest

import numpy as np

from hdlib.space import Space, Vector
from hdlib.arithmetic import bundle, bind, permute
from hdlib.parser import load_dataset, split_dataset


def read_params():
    """
    Read and test input arguments

    :return:        The ArgumentParser object
    """

    p = ap.ArgumentParser(
        prog="chopin2",
        description="chopin2 powered by hdlib",
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=os.path.abspath,
        required=True,
        help="Path to the input matrix",
    )
    p.add_argument(
        "--fieldsep",
        type=str,
        default="\t",
        help="Input field separator",
    )
    p.add_argument(
        "--dimensionality",
        type=int,
        default=10000,
        help="Vectors dimensionality",
    )
    p.add_argument(
        "--levels",
        type=int,
        required=True,
        help="Number of HD levels",
    )
    p.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds for cross-validating the classification model",
    )
    p.add_argument(
        "--retrain",
        type=int,
        default=10,
        help="Number of retraining iterations",
    )
    p.add_argument(
        "-v",
        "--version",
        action="version",
        version="chopin2 (hdlib) version {} ({})".format(__version__, __date__),
        help="Print the tool version and exit",
    )

    return p.parse_args()


def chopin2():
    """
    The ML model implemented in chopin2 is described in the following scientific paper:

    Cumbo F, Cappelli E, Weitschek E.
    A brain-inspired hyperdimensional computing approach for classifying massive dna methylation data of cancer
    Algorithms. 2020 Sep 17;13(9):233
    https://doi.org/10.3390/a13090233
    """

    # Load command line parameters
    args = read_params()

    # Load the input matrix
    print("Loading dataset")
    _, features, content, classes, min_value, max_value = load_dataset(args.input, sep=args.fieldsep)

    # Initialize the hyperdimensional space
    # Use bipolar vectors
    space = Space(size=args.dimensionality, vtype="bipolar")

    # Build HD levels
    print("Building HD levels")

    index_vector = range(args.dimensionality)
    next_level = int((args.dimensionality / 2 / args.levels))
    change = int(args.dimensionality / 2)

    # Also define the interval level list
    level_list = list()

    gap = (max_value - min_value) / args.levels

    for level_count in range(args.levels):
        level = "level_{}".format(level_count)

        if level_count == 0:
            base = np.full(args.dimensionality, -1)
            to_one = np.random.RandomState(seed=0).permutation(index_vector)[:change]

        else:
            to_one = np.random.RandomState(seed=0).permutation(index_vector)[:next_level]

        for index in to_one:
            base[index] = base[index] * -1

        vector = Vector(
            name=level,
            size=args.dimensionality,
            vtype="bipolar",
            vector=copy.deepcopy(base)
        )

        space.insert(vector)

        if level_count > 0:
            space.link(name1="level_{}".format(level_count - 1), name2=level)

        right_bound = min_value + level_count * gap

        if level_count == 0:
            left_bound = right_bound

        else:
            left_bound = min_value + (level_count - 1) * gap

        level_list.append((left_bound, right_bound))

    # Encode all data points
    print("Encoding data points")

    for point_position, point in enumerate(content):
        sum_vector = None

        for value_position, value in enumerate(point):

            if value == min_value:
                level_count = 0

            elif value == max_value:
                level_count = args.levels - 1

            else:
                for level_position in range(len(level_list)):
                    left_bound, right_bound = level_list[level_position]

                    if left_bound <= value and right_bound > value:
                        level_count = level_position

                        break

            level_vector = space.get(names=["level_{}".format(level_count)])[0]

            roll_vector = permute(level_vector, rotate_by=value_position)

            if sum_vector is None:
                sum_vector = roll_vector

            else:
                sum_vector = bundle(sum_vector, roll_vector)

        # Add the hyperdimensional representation of the data point to the space
        sum_vector.name = "point_{}".format(point_position)
        space.insert(sum_vector)

        # Tag vector with its class label
        space.add_tag(name=sum_vector.name, tag=classes[point_position])

    # Split dataset in folds
    folds = split_dataset(len(content), args.folds)

    # Take track of the accuracy for each classification model
    accuracies = list()

    for pos1, test_positions in enumerate(folds):
        print("Processing fold {}".format(pos1))

        training_points = ["point_{}".format(pos) for pos in range(len(content)) if pos not in test_positions]

        class_vectors = list()

        for class_pos, class_label in enumerate(sorted(classes)):
            # Get training vectors for the current class label
            class_points = [vector for vector in space.get(names=training_points) if class_label in vector.tags]

            # Build the vector representations of the current class
            class_vector = None

            for vector in class_points:
                if class_vector is None:
                    class_vector = vector

                else:
                    class_vector = bundle(class_vector, vector)

            class_vector.name = "class_{}_fold_{}".format(class_pos, pos1)

            class_vector.tags.add(class_label)

            # Add the class vector to the space
            space.insert(class_vector)

            class_vectors.append(class_vector)

        # Compute the distance between the test points and the hyperdimensional representations of classes
        test_points = ["point_{}".format(pos) for pos in test_positions]

        test_vectors = space.get(names=test_points)

        # Take track of the best accuracy while retraining
        best_accuracy = np.NINF

        # Retrain model
        retraining_iter = 0

        retraining_class_vectors = copy.deepcopy(class_vectors)

        # Take track of the predictions in the last retraining iteration 
        last_predictions = dict()

        retrain = args.retrain

        while retrain + 1 > 0:
            if retraining_iter > 0:
                for test_point in last_predictions:
                    # In case the test point has been wrongly predicted
                    test_position = int(test_point.split("_")[1])

                    if last_predictions[test_point] != classes[test_position]:
                        for class_vector in retraining_class_vectors:
                            if last_predictions[test_point] in class_vector.tags:
                                class_vector.vector = class_vector.vector - space.get(names=[test_point])[0].vector

                            if classes[test_position] in class_vector.tags:
                                class_vector.vector = class_vector.vector + space.get(names=[test_point])[0].vector

            # Count correctly classified points
            correctly_classified = 0

            # Also count the number of wrongly classified points for computing the error rate
            wrongly_predicted = 0

            for test_vector in test_vectors:
                closest_class = None
                closest_dist = np.NINF

                for class_vector in retraining_class_vectors:
                    distance = test_vector.dist(class_vector, method="cosine")

                    if closest_class is None:
                        closest_class = list(class_vector.tags)[0]
                        closest_dist = distance

                    else:
                        if distance > closest_dist:
                            closest_class = list(class_vector.tags)[0]
                            closest_dist = distance

                test_position = int(test_vector.name.split("_")[1])

                if classes[test_position] == closest_class:
                    # Correctly classified
                    correctly_classified += 1
                
                else:
                    wrongly_predicted += 1

                last_predictions[test_vector.name] = closest_class

            # Compute the accuracy
            accuracy = correctly_classified / len(test_vectors)

            # Compute the error rate
            error_rate = wrongly_predicted / len(last_predictions)

            print("\tAccuracy: {:.2f}  Error: {:.2f}".format(accuracy, error_rate))

            if accuracy > best_accuracy:
                best_accuracy = accuracy

            retrain -= 1

            retraining_iter += 1

        # Take track of the accuracy
        accuracies.append(best_accuracy)
    
    # Compute the classification model accuracy as the average of the accuracies
    accuracy = sum(accuracies) / len(accuracies)

    print("Accuracy: {:.2f}".format(accuracy))


if __name__ == "__main__":
    t0 = time.time()

    chopin2()

    t1 = time.time()
    print("Total elapsed time: {}s".format(int(t1 - t0)))
