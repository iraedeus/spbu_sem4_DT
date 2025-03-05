"""Module for splitting datasets into training and testing subsets.

This module provides a function `train_test_split` to divide a dataset
(features and labels) into separate training and testing sets.
"""

import random

from ml_spbu.homework_2.annotations import Point


def train_test_split(
    X: list[Point], y: list[int], test_size: float = 0.2, random_state: int = 42
) -> tuple[list[Point], list[Point], list[int], list[int]]:
    """Split arrays or matrices into random train and test subsets.

    This function is used to split datasets into training and testing sets.
    It shuffles the dataset and then splits it according to the `test_size`
    parameter.

    Args:
        X (list[Point]): List of feature points. This is the first dataset to split.
        y (list[int]): List of labels corresponding to the feature points in X.
            Must have the same length as X.
        test_size (float, optional): Represents the proportion of the dataset to include in
            the test split. Should be between 0.0 and 1.0. Default is 0.2.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls. Default is 42.

    Returns:
        tuple[list[Point], list[Point], list[int], list[int]]: A tuple containing four lists:
            - X_train (list[Point]): Training subset of the feature points.
            - X_test (list[Point]): Testing subset of the feature points.
            - y_train (list[int]): Training subset of the labels.
            - y_test (list[int]): Testing subset of the labels.

    Examples:
        >>> X = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
        >>> y = [0, 1, 0, 1, 0]
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
        >>> X_train
        [(9, 10), (7, 8), (5, 6)]
        >>> X_test
        [(1, 2), (3, 4)]
        >>> y_train
        [0, 1, 0]
        >>> y_test
        [0, 1]
    """

    if not (0 <= test_size <= 1):
        raise ValueError("Test size must be from 0 to 1.")

    X_labels = [(x, label) for x, label in zip(X, y)]
    random.seed(random_state)
    random.shuffle(X_labels)

    separator = round(len(X_labels) * test_size)

    X_train, y_train = [], []
    for x, label in X_labels[separator:]:
        X_train.append(x)
        y_train.append(label)

    X_test, y_test = [], []
    for x, label in X_labels[:separator]:
        X_test.append(x)
        y_test.append(label)

    return X_train, X_test, y_train, y_test
