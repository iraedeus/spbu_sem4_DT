"""Module implementing the K-Nearest Neighbors Classifier.

This module defines the KNNClassifier class, which uses a KD-Tree for efficient
nearest neighbor searches. It supports different distance metrics and allows
for classification tasks based on the majority class among the k-nearest neighbors.
"""

from ml_spbu.homework_2.annotations import CallableMetric, Point
from ml_spbu.homework_2.exceptions import NotFittedError
from ml_spbu.homework_2.KDTree.kd_tree import KDTree


class KNNClassifier:
    """K-Nearest Neighbors Classifier using KD-Tree.

    This class implements a K-Nearest Neighbors classifier that utilizes a KD-Tree
    for efficient neighbor searching. It can be configured with different distance
    metrics and leaf sizes for the KD-Tree.

    Attributes:
        _k (int): The number of neighbors to consider.
        _leaf_size (int): The leaf size for the KD-Tree.
        _metric (CallableMetric): The distance metric function.
        _kd_tree (KDTree): The KD-Tree fitted on the training data.
        _labels (dict[Point, int]): A dictionary mapping training points to their labels.
    """

    def __init__(self, k: int, leaf_size: int, metric: CallableMetric):
        """Initialize KNNClassifier with specified parameters.

        Args:
            k (int): The number of neighbors to use for classification.
            leaf_size (int): The leaf size for constructing the KD-Tree.
            metric (CallableMetric): The distance metric function to use.
                It should be a callable that takes two points and returns a float distance.
        """

        self._k = k
        self._leaf_size = leaf_size
        self._metric = metric

    def fit(self, X: list[Point], Y: list[int]):
        """Fit the KNN classifier to the training data.

        This method builds a KD-Tree from the training data and stores the
        corresponding labels. The KD-Tree is used for efficient nearest neighbor
        queries during prediction.

        Args:
            X (list[Point]): List of training data points.
            Y (list[int]): List of labels corresponding to the training data points.
        """

        self._kd_tree = KDTree(X, self._leaf_size, self._metric)
        self._labels = {x: y for x, y in zip(X, Y)}

    def predict_proba(self, X: list[Point]) -> list[tuple[float, float]]:
        """Predict class probabilities for the input data.

        For each point in X, this method finds the k-nearest neighbors in the
        training data using the KD-Tree and calculates the probability of
        belonging to each class (0 and 1) based on the class distribution
        among the neighbors.

        Args:
            X (list[Point]): List of data points for which to predict probabilities.

        Returns:
            list[tuple[float, float]]: List of tuples, where each tuple contains
                the probability of class 0 and class 1 for the corresponding
                input point.

        Raises:
            NotFittedError: If the `fit` method has not been called before `predict_proba`.
        """

        if not hasattr(self, "_kd_tree"):
            raise NotFittedError()

        output = []
        results = self._kd_tree.query(X, self._k)

        for result in results:
            negatives = sum(self._labels[neighbor] == 0 for neighbor in result)
            positives = sum(self._labels[neighbor] == 1 for neighbor in result)
            output.append((negatives / len(result), positives / len(result)))

        return output

    def predict(self, X: list[Point]) -> list[int]:
        """Predict class labels for the input data.

        This method predicts the class label for each point in X based on the
        class probabilities calculated by `predict_proba`. The class with the
        higher probability is assigned as the predicted label.

        Args:
            X (list[Point]): List of data points for which to predict labels.

        Returns:
            list[int]: List of predicted class labels (0 or 1) for each input point.
        """

        output = []
        labels = self.predict_proba(X)

        for proba in labels:
            output.append(int(proba[1] > proba[0]))

        return output
