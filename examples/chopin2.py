#!/usr/bin/env python3
"""
Implementation of chopin2 ML model with hdlib
https://github.com/cumbof/chopin2
"""

__author__ = ("Fabio Cumbo (fabio.cumbo@gmail.com)")

__version__ = "1.1.0"
__date__ = "Jun 14, 2023"

import argparse as ap
import os
import statistics
import time

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tabulate import tabulate

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
        "--feature-selection",
        type=str.lower,
        choices=["forward", "backward"],
        dest="feature_selection",
        help="Run the feature selection and report a ranking of features based on their importance",
    )
    p.add_argument(
        "--retrain",
        type=int,
        default=0,
        help="Number of retraining iterations",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Threshold on the accuracy score. Used in conjunction with --feature-selection",
    )
    p.add_argument(
        "--uncertainty",
        type=float,
        default=5.0,
        help="Uncertainty percentage. Used in conjunction with --feature-selection",
    )
    p.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Make it parallel when possible",
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

    if args.feature_selection and args.kfolds == 0:
        raise Exception("The --feature-selection option can only be used in conjunction with --kfolds")

    if args.kfolds == 0 and args.test_percentage == 0.0:
        raise Exception(
            (
                "Please use --kfold if you want to cross-validate your model. "
                "Otherwise, use --test-percentage to specify the percentage of data points for the test set. "
                "Use --help for other options"
            )
        )

    if args.nproc < 1:
        args.nproc = os.cpu_count()

    # Load the input matrix
    print("Loading dataset")
    
    _, features, content, classes = load_dataset(args.input, sep=args.fieldsep)

    print("Points: {}; Features {}; Classes {}".format(len(content), len(features), len(set(classes))))

    # Initialise the model
    print("Building the HD model")
    
    model = Model(size=args.dimensionality, levels=args.levels, vtype="bipolar")

    if not args.feature_selection:
        # Fit the model
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
            print("Percentage-split: training={}% test={}%".format(100.0 - args.test_percentage, args.test_percentage))

            test_indices = percentage_split(len(content), args.test_percentage)

            predictions = [
                model.predict(
                    test_indices,
                    distance_method="cosine",
                    retrain=args.retrain,
                    stop_if_worse=True
                )
            ]

        # For each prediction, compute the accuracy, f1, precision, and recall
        accuracy_scores = list()
        f1_scores = list()
        precision_scores = list()
        recall_scores = list()

        retraining_iterations = list()

        for y_indices, y_pred, retrainings in predictions:
            y_true = [label for position, label in enumerate(classes) if position in y_indices]

            accuracy_scores.append(accuracy_score(y_true, y_pred))
            f1_scores.append(f1_score(y_true, y_pred), average="micro")
            precision_scores.append(precision_score(y_true, y_pred), average="micro")
            recall_scores.append(recall_score(y_true, y_pred), average="micro")

            retraining_iterations.append(retrainings)

        print("Accuracy: {:.2f}".format(statistics.mean(accuracy_scores)))
        print("F1-Score: {:.2f}".format(statistics.mean(f1_scores)))
        print("Precision: {:.2f}".format(statistics.mean(precision_scores)))
        print("Recall: {:.2f}".format(statistics.mean(recall_scores)))

        print("Retraining iterations: {}".format(statistics.mean(retraining_iterations)))

    else:
        print("Selecting features.. This may take a while\n")

        # Run the feature selection
        importance, scores, top_importance = model.stepwise_regression(
            content,
            features,
            classes,
            method=args.feature_selection,
            cv=args.kfolds,
            distance_method="cosine",
            retrain=args.retrain,
            n_jobs=args.nproc,
            metric="accuracy",
            threshold=args.threshold,
            uncertainty=args.uncertainty
        )

        # Print features in ascending order on their score
        table = [["Feature", "Importance"]]

        for feature in sorted(importance.keys(), key=lambda f: importance[f]):
            table.append([feature, importance[feature]])

        print(tabulate(table, headers="firstrow", tablefmt="simple"))

        print()

        # Also print the score for each importance level
        table = [["Importance", "Score (accuracy)"]]

        for imp in sorted(scores.keys()):
            table.append([imp, scores[imp]])

        print(tabulate(table, headers="firstrow", tablefmt="simple"))

        print("\nAccuracy: {:.2f}".format(scores[top_importance]))

if __name__ == "__main__":
    t0 = time.time()

    chopin2()

    t1 = time.time()
    print("Total elapsed time: {}s".format(int(t1 - t0)))
