from ml_spbu.homework_2.annotations import Point


def minkowski_distance(p1: Point, p2: Point, p: float) -> float:
    if len(p1) != len(p2):
        raise AttributeError("The dimensions of the points must be the same.")
    if p < 1:
        raise AttributeError("The p attribute must be greater than or equal to 1")

    return (sum(abs(p1[axis] - p2[axis]) ** p for axis in range(len(p1)))) ** (1 / p)
