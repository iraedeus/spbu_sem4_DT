"""Module for calculating Manhattan distance between two points.

This module provides a function to compute the Manhattan distance, also known
as L1 distance or taxicab geometry distance, between two points.
"""

from ml_spbu.homework_2.annotations import Point


def manhattan_distance(p1: Point, p2: Point) -> float:
    """Calculate the Manhattan distance between two points.

    Manhattan distance is the sum of the absolute differences of their
    Cartesian coordinates. In a 2D plane, if point p1 has coordinates (x1, y1)
    and point p2 has coordinates (x2, y2), the Manhattan distance is
    |x1 - x2| + |y1 - y2|.

    Args:
        p1 (Point): The first point, represented as a tuple of coordinates.
        p2 (Point): The second point, represented as a tuple of coordinates.

    Returns:
        float: The Manhattan distance between the two points.

    Raises:
        AttributeError: If the dimensions of the two points are not the same.

    Examples:
        >>> manhattan_distance((1, 2), (4, 6))
        7.0
        >>> manhattan_distance((0, 0, 0), (3, 4, 0))
        7.0
    """

    if len(p1) != len(p2):
        raise AttributeError("The dimensions of the points must be the same.")
    return sum(abs(p1[axis] - p2[axis]) for axis in range(len(p1)))
