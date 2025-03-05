from collections.abc import Callable

Point = tuple[float, ...]
CallableMetric = Callable[[Point, Point], float]
