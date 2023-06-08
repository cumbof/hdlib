#!/usr/bin/env python3
"""
Implementation of chopin2 ML model with hdlib
https://github.com/cumbof/chopin2
"""

__author__ = ("Fabio Cumbo (fabio.cumbo@gmail.com)")

__version__ = "1.1.0"
__date__ = "Jun 8, 2023"

import argparse as ap
import os
import time

from sklearn.metrics import accuracy_score

from hdlib.parser import load_dataset, percentage_split
from hdlib.model import Model


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
        "--kfolds",
        type=int,
        default=0,
        help="Number of folds for cross-validating the classification model",
    )
    p.add_argument(
        "--test-percentage",
        type=float,
        default=0.0,
        dest="test_percentage",
        help="Percentage of data points for defining the test set",
    )
    p.add_argument(
        "--retrain",
        type=int,
        default=0,
        help="Number of retraining iterations",
    )
    p.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Make it parallel",
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

    if args.kfolds == 0 and args.test_percentage == 0.0:
        raise Exception(
            (
                "Please use --kfold if you want to cross-validate your model. "
                "Otherwise, use --test-percentage to specify the percentage of data points for the test set. "
                "Use --help for other options"
            )
        )

    # Load the input matrix
    print("Loading dataset")
    
    _, _, content, classes = load_dataset(args.input, sep=args.fieldsep)

    # Initialise the model
    print("Building the HD model")
    
    model = Model(size=args.dimensionality, levels=args.levels, vtype="bipolar")

    model.fit(content, classes)

    if args.kfolds > 0:
        # Predict in cross-validation
        print("Cross-validating model with {} folds".format(args.kfolds))

        predictions = model.cross_val_predict(
            content,
            classes,
            cv=args.kfolds,
            distance_method="cosine",
            retrain=args.retrain,
            n_jobs=args.nproc
        )

    elif args.test_percentage > 0.0:
        # Predict with a percentage-split
        test_indices = percentage_split(len(content), args.test_percentage)

        predictions = [
            model.predict(
                test_indices,
                distance_method="cosine",
                retrain=args.retrain
            )
        ]

    # For each prediction, compute the accuracy
    accuracy_scores = list()

    for y_indices, y_pred in predictions:
        y_true = [label for position, label in classes if position in y_indices]

        accuracy_scores.append(accuracy_score(y_true, y_pred))

    print("Accuracy: {:.2f}".format(sum(accuracy_scores) / len(accuracy_scores)))


if __name__ == "__main__":
    t0 = time.time()

    chopin2()

    t1 = time.time()
    print("Total elapsed time: {}s".format(int(t1 - t0)))
