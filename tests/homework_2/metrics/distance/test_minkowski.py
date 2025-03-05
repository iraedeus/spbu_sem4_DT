import pytest

from ml_spbu.homework_2.metrics.distance.minkowski import minkowski_distance


class TestMinkowskiDistance:
    @pytest.mark.parametrize(
        "point1, point2, p_value, expected_distance",
        [
            ((0, 0), (0, 0), 2, 0.0),
            ((1, 0), (0, 0), 2, 1.0),
            ((1, 1), (0, 0), 2, 1.41421356),
            ((1, 1), (0, 0), 1, 2.0),
            ((3, 4), (0, 0), 2, 5.0),
            ((-1, -1), (0, 0), 2, 1.41421356),
            ((1.5, 2.5), (0, 0), 2, 2.91547595),
            ((1, 1), (0, 0), 3, 1.25992104989),
            ((2, 3), (1, 2), 1.4, 1.64067071202),
        ],
    )
    def test_basic_minkowski_distance(self, point1, point2, p_value, expected_distance):
        assert minkowski_distance(point1, point2, p_value) == pytest.approx(expected_distance, abs=1e-8)

    def test_points_same_reference_minkowski(self):
        p = (1, 2)
        assert minkowski_distance(p, p, 2) == 0.0

    @pytest.mark.parametrize(
        "point1, point2",
        [
            ((1, 2), (1, 2, 3)),
            ((1,), (1, 2)),
        ],
    )
    def test_minkowski_distance_different_dimensions(self, point1, point2):
        with pytest.raises(AttributeError):
            minkowski_distance(point1, point2, 2)

    def test_minkowski_distance_invalid_p_value(self):
        point1 = (1, 1)
        point2 = (0, 0)
        with pytest.raises(AttributeError):
            minkowski_distance(point1, point2, 0.9)
