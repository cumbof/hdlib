#!/usr/bin/env python3
"""Implementation of chopin2 ML model with hdlib."""

__author__ = ("Fabio Cumbo (fabio.cumbo@gmail.com)")

__version__ = "1.1.0"
__date__ = "Jul 16, 2023"

import argparse as ap
import errno
import os
import statistics
import time
from typing import List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from tabulate import tabulate

from hdlib.model import ClassificationModel


def read_params():
    """Read and test the input arguments.

    Returns
    -------
    argparse.ArgumentParser
        The ArgumentParser object
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


def load_dataset(filepath: os.path.abspath, sep: str="\t") -> Tuple[List[str], List[List[float]], List[str]]:
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


def chopin2():
    """Domain-agnostic supervised learning with hyperdimensional computing.
    
    Notes
    -----
    chopin2 is available in GitHub at https://github.com/cumbof/chopin2 and its ML model is described
    in the following scientific article:

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
                "Please use --kfolds if you want to cross-validate your model. "
                "Otherwise, use --test-percentage to specify the percentage of data points for the test set. "
                "Use --help for other options"
            )
        )

    if args.nproc < 1:
        args.nproc = os.cpu_count()

    # Load the input matrix
    print("Loading dataset")
    
    _, features, content, classes = load_dataset(args.input, sep=args.fieldsep)

    class_labels = sorted(list(set(classes)))

    print("Points: {}; Features {}; Classes {}".format(len(content), len(features), len(class_labels)))

    # Initialise the model
    print("Building the HD model")
    
    model = ClassificationModel(size=args.dimensionality, levels=args.levels, vtype="bipolar")

    if args.feature_selection:
        print("Selecting features.. This may take a while\n")

        # Run the feature selection
        importances, scores, best_importance, count_models = model.stepwise_regression(
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

        for feature in sorted(importances.keys(), key=lambda f: importances[f]):
            table.append([feature, importances[feature]])

        print(tabulate(table, headers="firstrow", tablefmt="simple"))

        print()

        # Also print the score for each importance level
        table = [["Importance", "Score (accuracy)"]]

        for imp in sorted(scores.keys()):
            table.append([imp, "{:.4f}".format(scores[imp])])

        print(tabulate(table, headers="firstrow", tablefmt="simple"))

        print("\nBest importance: {}".format(best_importance))

        # Select features based on the best importance
        if args.feature_selection == "backward":
            selected_features = [feature for feature in importances if importances[feature] <= best_importance]

        elif args.feature_selection == "forward":
            selected_features = [feature for feature in importances if importances[feature] >= best_importance]

        print("Selected features: {}\n".format(len(selected_features)))

        # Also print the score for each importance level
        table = [["Features"]]

        for feature in selected_features:
            table.append([feature])

        print(tabulate(table, headers="firstrow", tablefmt="simple"))

        print("\nTotal ML models: {}".format(count_models))

        # Replace features with their positions in the original list of features
        selected_features = [features.index(feature) for feature in selected_features]

        print("\nBuilding a model with the selected features only")

        # Reshape content by considering the selected features only
        content = [[value for position, value in enumerate(sample) if position in selected_features] for sample in content]

    # Fit the model
    model.fit(content, classes)

    if args.kfolds > 0:
        # Predict in cross-validation
        print("Cross-validating model with {} folds\n".format(args.kfolds))

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
        print("Percentage-split: training={}% test={}%\n".format(100.0 - args.test_percentage, args.test_percentage))

        test_indices = percentage_split(classes, args.test_percentage)

        predictions = [
            model.predict(
                test_indices,
                distance_method="cosine",
                retrain=args.retrain
            )
        ]

    # For each prediction, compute the Accuracy, F1, Precision, Recall, AUC
    accuracy_scores = list()
    f1_scores = list()
    precision_scores = list()
    recall_scores = list()
    mcc_scores = list()

    retraining_iterations = list()
    error_rates = list()

    print("Class labels: {}\n".format(class_labels))

    for fold, (y_indices, y_pred, retrainings, error_rate, _) in enumerate(predictions):
        # Produce the confusion matrix for each fold
        print("Fold {} ({} retrainings)".format(fold + 1, retrainings))
        retraining_iterations.append(retrainings)

        print("\tError rate: {}".format(error_rate))
        error_rates.append(error_rate)

        y_true = [classes[position] for position in y_indices]

        fold_accuracy = accuracy_score(y_true, y_pred)
        print("\tAccuracy: {}".format("{:.4f}".format(fold_accuracy)))
        accuracy_scores.append(fold_accuracy)

        fold_f1 = f1_score(y_true, y_pred, average="weighted")
        print("\tF1: {}".format("{:.4f}".format(fold_f1)))
        f1_scores.append(fold_f1)

        fold_precision = precision_score(y_true, y_pred, average="weighted")
        print("\tPrecision: {}".format("{:.4f}".format(fold_precision)))
        precision_scores.append(fold_precision)

        fold_recall = recall_score(y_true, y_pred, average="weighted")
        print("\tRecall: {}".format("{:.4f}".format(fold_recall)))
        recall_scores.append(fold_recall)

        fold_mcc = matthews_corrcoef(y_true, y_pred)
        print("\tMatthews Corr. Coeff.: {}".format("{:.4f}".format(fold_mcc)))
        mcc_scores.append(fold_mcc)

        print("\tConfusion Matrix:")
        if len(class_labels) > 2:
            for row in confusion_matrix(y_true, y_pred, labels=class_labels):
                print("\t\t{}".format(row))

        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=class_labels).ravel()

            print("\t\tTN: {}".format(tn))
            print("\t\tFP: {}".format(fp))
            print("\t\tFN: {}".format(fn))
            print("\t\tTP: {}".format(tp))

        print()

    for score_label, score_values in zip(
        ["Retrainings", "Error Rate", "Accuracy", "F1", "Precision", "Recall", "Matthews Corr. Coeff."],
        [retraining_iterations, error_rates, accuracy_scores, f1_scores, precision_scores, recall_scores, mcc_scores]
    ):
        print(
            "{}: {:.4f}{}".format(
                score_label,
                statistics.mean(score_values),
                " ({:.4f} stdev)".format(statistics.stdev(score_values)) if len(score_values) > 1 else ""
            )
        )


if __name__ == "__main__":
    t0 = time.time()

    chopin2()

    t1 = time.time()
    print("Total elapsed time: {}s".format(int(t1 - t0)))
