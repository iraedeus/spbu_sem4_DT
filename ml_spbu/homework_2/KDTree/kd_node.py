from math import inf
from statistics import median

from ml_spbu.homework_2.annotations import Point


class KDNode:
    def __init__(self, X: list[Point], leaf_size: int):
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
