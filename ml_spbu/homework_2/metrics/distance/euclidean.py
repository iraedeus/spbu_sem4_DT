"""Module for calculating Euclidean distance between two points.

This module provides a function to compute the Euclidean distance, a measure
of the straight-line distance between two points in Euclidean space.
"""

from math import sqrt

from ml_spbu.homework_2.annotations import Point


def euclidean_distance(p1: Point, p2: Point) -> float:
    """Calculate the Euclidean distance between two points.

    Euclidean distance is the straight-line distance between two points in
    Euclidean space. It is calculated as the square root of the sum of the
    squared differences of their coordinates.

    Args:
        p1 (Point): The first point, represented as a tuple of coordinates.
        p2 (Point): The second point, represented as a tuple of coordinates.

    Returns:
        float: The Euclidean distance between the two points.

    Raises:
        AttributeError: If the dimensions of the two points are not the same.

    Examples:
        >>> euclidean_distance((1, 2), (4, 6))
        5.0
        >>> euclidean_distance((0, 0, 0), (3, 4, 0))
        5.0
    """

    if len(p1) != len(p2):
        raise AttributeError("The dimensions of the points must be the same.")
    return sqrt(sum((p1[axis] - p2[axis]) ** 2 for axis in range(len(p1))))
