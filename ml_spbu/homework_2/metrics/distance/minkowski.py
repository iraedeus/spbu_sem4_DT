"""Module for calculating Minkowski distance between two points.

This module provides a function to compute the Minkowski distance, a generalized
metric in normed vector space which is a generalization of both the Euclidean
distance and the Manhattan distance.
"""

from ml_spbu.homework_2.annotations import Point


def minkowski_distance(p1: Point, p2: Point, p: float) -> float:
    """Calculate the Minkowski distance between two points.

    The Minkowski distance is a generalization of the Euclidean and Manhattan
    distances. When p=2, it is equivalent to the Euclidean distance. When p=1,
    it is equivalent to the Manhattan distance.

    Args:
        p1 (Point): The first point, represented as a tuple of coordinates.
        p2 (Point): The second point, represented as a tuple of coordinates.
        p (float): The order of the Minkowski distance. Must be greater than or equal to 1.

    Returns:
        float: The Minkowski distance between the two points.

    Raises:
        AttributeError:
            - If the dimensions of the two points are not the same.
            - If the value of 'p' is less than 1.

    Examples:
        >>> minkowski_distance((1, 2), (4, 6), 2)  # Euclidean distance (p=2)
        5.0
        >>> minkowski_distance((1, 2), (4, 6), 1)  # Manhattan distance (p=1)
        7.0
        >>> minkowski_distance((0, 0, 0), (3, 4, 0), 3)
        5.848035476425731
    """

    if len(p1) != len(p2):
        raise AttributeError("The dimensions of the points must be the same.")
    if p < 1:
        raise AttributeError("The p attribute must be greater than or equal to 1")

    return (sum(abs(p1[axis] - p2[axis]) ** p for axis in range(len(p1)))) ** (1 / p)
