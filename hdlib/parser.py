"""Utility to parse input files."""

import errno
import os
import random
from typing import List, Tuple

import numpy as np


def load_dataset(
    filepath: os.path.abspath,
    sep: str="\t"
) -> Tuple[List[str], List[List[float]], List[str]]:
    """Load the input dataset.

    Parameters
    ----------
    filepath : str
        Path to the input dataset.
    sep : str
        Filed separator for the input dataset.

    Returns
    -------
    tuple
        A tuple with a list of sample IDs, a list of features, a list of lists with the
        actual numerical data (floats), and a list with class labels.
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
    """Given a number of data points and the number of folds, split a dataset into different folds.

    Parameters
    ----------
    points : int
        Number of data points in the input dataset.
    folds : int
        Number of folds.

    Returns
    -------
    list
        A list of lists. Every list is a fold with the indices of data points in the original dataset.

    Raises
    ------
    ValueError
        If the number of folds is greater than the number of data points.

    Examples
    --------
    >>> from hdlib.parser import kfolds_split
    >>> kfolds_split(10, 3)
    [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]

    Considering a dataset with 10 data points, split them into 3 folds.
    """

    if folds > points:
        raise ValueError("The number of folds cannot exceed the number of data points")

    data_points = list(range(points))

    return [data_points[i::folds] for i in range(folds)]


def percentage_split(points: int, percentage: float, seed: int=0) -> List[int]:
    """Given a number of data points and a percentage number, split a dataset and report the indices of the of data points.

    Parameters
    ----------
    points : int
        Number of data points in the input dataset.
    percentage : float
        Percentage of points to split out of the original dataset.
    seed : int
        Random seed for reproducing the same results.

    Returns
    -------
    list
        A list with the indices of selected points.

    Raises
    ------
    ValueError
        - if the input `percentage` is lower than or equal to 0.0 or greater than 100.0;
        - if the input `seed` is not an integer number.

    Examples
    --------
    >>> from hdlib.parser import percentage_split
    >>> percentage_split(10, 20.0, seed=0)
    [6, 9]

    Consider a dataset with 10 data points, select 20% of the points (2 points in this case),
    and report their indices in the original dataset.
    """

    if percentage <= 0.0 or percentage > 100.0:
        raise ValueError("Percentage must be greater than 0 and lower than or equal to 100")

    if not isinstance(seed, int):
        raise ValueError("The input seed must be an integer number")

    select_points = percentage * points / 100.0

    random.seed(seed)

    return random.sample(list(range(points)), int(select_points))
