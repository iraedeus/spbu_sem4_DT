"""Module implementing a KD-Tree Node for efficient spatial data partitioning.

This module defines the KDNode class, which represents a node in a KD-Tree.
"""

from math import inf
from statistics import median

from ml_spbu.homework_2.annotations import Point


class KDNode:
    """Represents a node in a KD-Tree data structure.

    Each KDNode can be either a leaf node or an internal node.
    Internal nodes contain information about the splitting axis and median value,
    and have references to left and right child nodes. Leaf nodes contain a list
    of data points that fall within the node's region.

    Attributes:
        axis (int | None): The dimension index used for splitting at this node.
            None if it's a leaf node.
        axis_median (float | None): The median value along the splitting axis.
            None if it's a leaf node.
        is_leaf (bool): True if the node is a leaf node, False otherwise.
        left (KDNode | None): The left child node. None if it's a leaf node.
        right (KDNode | None): The right child node. None if it's a leaf node.
        X (list[Point] | None): List of data points contained in this leaf node.
            None if it's an internal node.
    """

    def __init__(self, X: list[Point], leaf_size: int):
        """Initialize a KDNode.

        Recursively constructs a KDNode. If the number of points in X is greater
        than `leaf_size`, it creates an internal node by choosing a splitting
        axis, finding the median along that axis, and splitting the data into
        left and right subsets. Otherwise, it creates a leaf node storing the points in X.

        Args:
            X (list[Point]): List of data points to be stored in this node and its children.
            leaf_size (int): The maximum number of data points a leaf node can contain.

        Raises:
            AttributeError: If `leaf_size` is not strictly positive.
        """

        if leaf_size <= 0:
            raise AttributeError("Leaf size must be strictly positive")

        self.axis: int | None = None
        self.axis_median: float | None = None
        self.is_leaf: bool = False
        self.left: KDNode | None = None
        self.right: KDNode | None = None

        # Initialization of children
        if len(X) > leaf_size:
            self.axis = self._choose_axis(X)
            axis_median, left_X, right_X = self._split(X, self.axis)

            self.axis_median = axis_median

            self.left = KDNode(left_X, leaf_size)
            self.right = KDNode(right_X, leaf_size)
        else:
            self.is_leaf = True
            self.X = X

    def _choose_axis(self, X: list[Point]) -> int:
        """Choose the axis with the largest spread for splitting.

        Selects the dimension along which the data points in X have the greatest
        range (difference between maximum and minimum values). This heuristic
        aims to create balanced KD-Trees.

        Args:
            X (list[Point]): List of data points to consider for axis selection.

        Returns:
            int: The index of the chosen axis (dimension).
        """

        output_axis = 0
        max_spread = -inf
        dim = len(X[0])

        for axis in range(dim):
            axis_min, axis_max = inf, -inf
            for x in X:
                axis_max = max(axis_max, x[axis])
                axis_min = min(axis_min, x[axis])

            spread = axis_max - axis_min
            if max_spread < spread:
                max_spread = spread
                output_axis = axis

        return output_axis

    def _split(self, X: list[Point], axis: int) -> tuple[float, list[Point], list[Point]]:
        """Split the data points based on the median along the given axis.

        Partitions the list of points X into two lists, `left_X` and `right_X`,
        based on whether their coordinate along the given `axis` is less than
        the median value along that axis. If either `left_X` or `right_X` is empty
        after the split, it performs a simple split by dividing X into two halves.

        Args:
            X (list[Point]): List of data points to be split.
            axis (int): The dimension index to split along.

        Returns:
            tuple[float, list[Point], list[Point]]:
                - axis_median (float): The median value along the splitting axis.
                - left_X (list[Point]): Points with coordinate less than the median on the axis.
                - right_X (list[Point]): Points with coordinate greater than or equal to the median on the axis.
        """

        on_axis = {x[axis] for x in X}
        axis_median = median(on_axis)

        left_X = []
        right_X = []

        for x in X:
            if x[axis] < axis_median:
                left_X.append(x)
            else:
                right_X.append(x)

        if not left_X or not right_X:
            left_X, right_X = X[: len(X) // 2], X[len(X) // 2 :]

        return axis_median, left_X, right_X
