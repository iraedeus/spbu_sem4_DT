"""Module for calculating the accuracy score for classification tasks.

This module provides a function to evaluate the accuracy of classification
predictions by comparing them against the true labels.
"""


def accuracy_score(y_pred: list[int], y_true: list[int]) -> float:
    """Calculate the accuracy score between predicted and true labels.

    Accuracy score is the fraction of correctly classified samples.

    Args:
        y_pred (list[int]): Predicted labels.
        y_true (list[int]): True labels.

    Returns:
        float: Accuracy score.

    Raises:
        AttributeError: If the lengths of `y_pred` and `y_true` are not the same.

    Examples:
        >>> accuracy_score([1, 0, 1, 0], [1, 1, 1, 0])
        0.75
        >>> accuracy_score([], [])
        1.0
    """

    if len(y_pred) != len(y_true):
        raise AttributeError("The lengths of arrays y_pred and y_true must be the same.")

    if len(y_pred) == len(y_true) == 0:
        return 1.0

    correct_answers = sum(pred == true for pred, true in zip(y_pred, y_true))
    return correct_answers / len(y_pred)
