from ml_spbu.homework_2.annotations import Point


def manhattan_distance(p1: Point, p2: Point) -> float:
    if len(p1) != len(p2):
        raise AttributeError("The dimensions of the points must be the same.")
    return sum(abs(p1[axis] - p2[axis]) for axis in range(len(p1)))
