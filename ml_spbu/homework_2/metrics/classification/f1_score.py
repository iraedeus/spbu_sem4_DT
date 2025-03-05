"""Module for calculating the F1 score for binary classification tasks.

This module provides a function to compute the F1 score, which is the harmonic mean
of precision and recall. It's particularly useful when dealing with imbalanced datasets.
"""


def f1_score(y_pred: list[int], y_true: list[int]) -> float:
    """Calculate the F1 score between predicted and true labels.

    The F1 score is the harmonic mean of precision and recall, providing a balanced
    measure of a test's accuracy. It is particularly useful in binary classification
    problems with imbalanced datasets.

    Args:
        y_pred (list[int]): Predicted labels (0 or 1).
        y_true (list[int]): True labels (0 or 1).

    Returns:
        float: F1 score. Returns 0 if there are no true positives.

    Raises:
        AttributeError: If the lengths of `y_pred` and `y_true` are not the same.

    Examples:
        >>> f1_score([1, 0, 1, 1, 0], [1, 1, 1, 0, 0])
        0.75
        >>> f1_score([0, 0, 0], [1, 1, 1])
        0.0
    """

    if len(y_pred) != len(y_true):
        raise AttributeError("The lengths of arrays y_pred and y_true must be the same.")

    true_positives = sum(pred == true == 1 for pred, true in zip(y_pred, y_true))
    false_positives = sum((pred == 1) and (pred != true) for pred, true in zip(y_pred, y_true))
    false_negatives = sum((pred == 0) and (pred != true) for pred, true in zip(y_pred, y_true))

    if true_positives == 0:
        return 0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return 2 * ((precision * recall) / (precision + recall))
