import pytest

from ml_spbu.homework_2.metrics.distance.euclidean import euclidean_distance


class TestEuclideanDistance:
    @pytest.mark.parametrize(
        "point1, point2, expected_distance",
        [
            ((0, 0), (0, 0), 0.0),
            ((1, 0), (0, 0), 1.0),
            ((0, 1), (0, 0), 1.0),
            ((1, 1), (0, 0), 1.41421356),
            ((-1, -1), (0, 0), 1.41421356),
            ((3, 4), (0, 0), 5.0),
            ((1.5, 2.5), (0, 0), 2.91547595),
        ],
    )
    def test_basic_euclidean_distance(self, point1, point2, expected_distance):
        assert euclidean_distance(point1, point2) == pytest.approx(expected_distance, abs=1e-8)

    def test_points_same_reference(self):
        p = (1, 2)
        assert euclidean_distance(p, p) == 0.0

    @pytest.mark.parametrize(
        "point1, point2",
        [
            ((1, 2), (1, 2, 3)),
            ((1,), (1, 2)),
        ],
    )
    def test_euclidean_distance_different_dimensions(self, point1, point2):
        with pytest.raises(AttributeError):
            euclidean_distance(point1, point2)
