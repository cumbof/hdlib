"""
Utility to parse input files
"""

import errno
import os
from typing import List, Tuple

import numpy as np


def load_dataset(
    filepath: os.path.abspath,
    sep: str="\t"
) -> Tuple[List[str], List[List[float]], List[str], float, float]:
    """
    Load the input dataset

    :param filepath:    Path to the input dataset
    :param sep:         Field separator
    :return:            The list of sample IDs, the list of features, the content as list of lists, 
                        and the list of classes, in addition to the minimum and maximum value
    """

    if not os.path.isfile(filepath):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)

    samples = list()
    content = list()
    classes = list()

    # Search for global minimum and maximum values
    global_min = np.Inf
    global_max = np.NINF

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

                local_min = min(row_data)

                if local_min < global_min:
                    global_min = local_min
                
                local_max = max(row_data)

                if local_max > global_max:
                    global_max = local_max

                # Take track of the class
                classes.append(line_split[-1])

    return samples, features, content, classes, global_min, global_max


def split_dataset(points: int, folds: int) -> List[List[int]]:
    """
    Given a number of data points and the number of folds, split a dataset into different folds

    :param points:  Number of data points
    :param folds:   Number of folds
    :return:        List of lists with the indeces of data points
    """

    data_points = list(range(points))

    return [data_points[i::folds] for i in range(folds)]
