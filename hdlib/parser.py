"""Utility to parse input files.

This module provides a set of utilities to parse input tables and split the dataset 
into training and test sets as a simple percentage split or cross validation."""

import errno
import os
from typing import List, Tuple

import numpy as np


def load_dataset(
    filepath: os.path.abspath,
    sep: str="\t"
) -> Tuple[List[str], List[List[float]], List[str]]:
    """Load the input numerical dataset.

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

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the input dataset does not contain number only.
    """

    if not os.path.isfile(filepath):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)

    samples = list()
    content = list()
    classes = list()

    try:
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

    except ValueError as e:
        raise Exception("The input dataset must contain numbers only!").with_traceback(e.__traceback__)

    return samples, features, content, classes


def percentage_split(labels: List[str], percentage: float, seed: int=0) -> List[int]:
    """Given list of classes as appear in the original dataset and a percentage number, split a dataset and 
    report the indices of the selected data points.

    Parameters
    ----------
    labels : list
        List of class labels as they appear in the original dataset.
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
    >>> labels = [1, 2, 2, 2, 1, 1, 1, 1, 2, 2]
    >>> percentage_split(labels, 20.0, seed=0)
    [6, 9]

    Consider a dataset with 10 data points, select 20% of the points (2 points in this case),
    and report their indices in the original dataset.
    """

    if percentage <= 0.0 or percentage > 100.0:
        raise ValueError("Percentage must be greater than 0 and lower than or equal to 100")

    if not isinstance(seed, int):
        raise ValueError("The input seed must be an integer number")

    unique_labels = list(set(labels))

    if len(unique_labels) < 2:
        raise ValueError("The list of class labels must contain at least two unique lables")

    rand = np.random.default_rng(seed=seed)

    selection = list()

    for label in unique_labels:
        # Get a specific percentage of the data points for a specific class
        select_points = percentage * labels.count(label) / 100.0

        # Retrieve the indices of the samples under a specific class in the original dataset
        indices = [pos for pos, val in enumerate(labels) if val == label]

        # Finally subsample the list of indices according to the specific percentage
        selection.extend([indices[i] for i in rand.choice(len(indices), int(select_points), replace=False)])

    return sorted(selection)
