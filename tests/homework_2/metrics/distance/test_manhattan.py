import pytest

from ml_spbu.homework_2.metrics.distance.manhattan import manhattan_distance


class TestManhattanDistance:
    @pytest.mark.parametrize(
        "point1, point2, expected_distance",
        [
            ((0, 0), (0, 0), 0.0),
            ((1, 0), (0, 0), 1.0),
            ((0, 1), (0, 0), 1.0),
            ((1, 1), (0, 0), 2.0),
            ((-1, -1), (0, 0), 2.0),
            ((3, 4), (0, 0), 7.0),
            ((1.5, 2.5), (0, 0), 4.0),
        ],
    )
    def test_basic_manhattan_distance(self, point1, point2, expected_distance):
        assert manhattan_distance(point1, point2) == pytest.approx(expected_distance, abs=1e-8)

    def test_points_same_reference_manhattan(self):
        p = (1, 2)
        assert manhattan_distance(p, p) == 0.0

    @pytest.mark.parametrize(
        "point1, point2",
        [
            ((1, 2), (1, 2, 3)),
            ((1,), (1, 2)),
        ],
    )
    def test_manhattan_distance_different_dimensions(self, point1, point2):
        with pytest.raises(AttributeError):
            manhattan_distance(point1, point2)
