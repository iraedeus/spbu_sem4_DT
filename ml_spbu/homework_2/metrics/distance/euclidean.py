from math import sqrt

from ml_spbu.homework_2.annotations import Point


def euclidean_distance(p1: Point, p2: Point) -> float:
    if len(p1) != len(p2):
        raise AttributeError("The dimensions of the points must be the same.")
    return sqrt(sum((p1[axis] - p2[axis]) ** 2 for axis in range(len(p1))))
