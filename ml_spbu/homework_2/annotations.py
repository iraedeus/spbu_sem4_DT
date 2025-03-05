"""Module defining type annotations for machine learning related types.

This module provides type aliases for points and callable metrics, to improve code readability and type checking.
"""

from collections.abc import Callable

Point = tuple[float, ...]
"""Type alias for a point in n-dimensional space.

Represents a data point as a tuple of floats, where each float is a coordinate
in a particular dimension. The number of elements in the tuple defines the
dimensionality of the point.

Examples:
    2D Point: (1.0, 2.5)
    3D Point: (0.5, 1.0, 3.2)
"""

CallableMetric = Callable[[Point, Point], float]
"""Type alias for a callable metric function.

Represents a function that calculates a distance or similarity metric
between two points. The function should accept two `Point` objects as
arguments and return a float representing the computed metric value.

The metric function should adhere to the following signature:
    Callable[[Point, Point], float]

Examples of metric functions:
    euclidean_distance, manhattan_distance, minkowski_distance
"""
