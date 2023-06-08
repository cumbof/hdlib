"""
Utility to parse input files
"""

import errno
import os
import random
from typing import List, Tuple

import numpy as np


def load_dataset(
    filepath: os.path.abspath,
    sep: str="\t"
) -> Tuple[List[str], List[List[float]], List[str]]:
    """
    Load the input dataset

    :param filepath:    Path to the input dataset
    :param sep:         Field separator
    :return:            The list of sample IDs, the list of features, the content as list of lists, and the list of classes
    """

    if not os.path.isfile(filepath):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)

    samples = list()
    content = list()
    classes = list()

    with open(filepath) as infile:
        # Trip the first and last column out
        features = infile.readline().rstrip().split(sep)[1:-1]

        for line in infile:
            line = line.strip()

            if line and not line.startswith("#"):
                line_split = line.split(sep)

                # Add sample ID
                samples.append(line_split[0])

                row_data = [float(value) for value in line_split[1:-1]]

                # Add row
                content.append(row_data)

                # Take track of the class
                classes.append(line_split[-1])

    return samples, features, content, classes


def kfolds_split(points: int, folds: int) -> List[List[int]]:
    """
    Given a number of data points and the number of folds, split a dataset into different folds

    :param points:  Number of data points
    :param folds:   Number of folds
    :return:        List of lists with the indices of data points
    """

    if folds > points:
        raise ValueError("The number of folds cannot exceed the number of data points")

    data_points = list(range(points))

    return [data_points[i::folds] for i in range(folds)]


def percentage_split(points: int, percentage: float) -> List[int]:
    """
    Given a number of data points and a percentage number, split a dataset in two and report
    the indices of the smallest set

    :param points:      Number of data points
    :param percentage:  Percentage split
    :return:            List with indices of the smallest set after split
    """

    if percentage <= 0.0 or percentage > 100.0:
        raise ValueError("Percentage must be greater than 0 and lower than or equal to 100")

    select_points = percentage * points / 100.0

    random.seed(0)

    return random.sample(list(range(points)), int(select_points))
