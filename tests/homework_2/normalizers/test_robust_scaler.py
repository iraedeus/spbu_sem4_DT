import pytest

from ml_spbu.homework_2.exceptions import NotFittedError
from ml_spbu.homework_2.normalizers.robust_scaler import RobustScaler


class TestRobustScaler:
    @pytest.mark.parametrize(
        "data, expected_scaled_data",
        [
            ([(1, 2), (3, 4), (5, 6), (7, 8)], [(-0.5, -0.5), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]),
            ([(0, 0), (10, 10), (20, 20), (30, 30)], [(-0.5, -0.5), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]),
        ],
    )
    def test_robust_scaler_basic(self, data, expected_scaled_data):
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data)

        assert len(scaled_data) == len(expected_scaled_data)
        for i in range(len(scaled_data)):
            assert len(scaled_data[i]) == len(expected_scaled_data[i])
            for j in range(len(scaled_data[i])):
                assert scaled_data[i][j] == pytest.approx(expected_scaled_data[i][j])

    @pytest.mark.parametrize(
        "data, expected_scaled_data",
        [
            ([(5, 5)], [(0.0, 0.0)]),
            (
                [(0, 0), (0, 0), (0, 0), (0, 0)],
                [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            ),  # Median 0, IQR 0. Scaled to 0.
            (
                [(10, 10), (10, 10), (10, 10), (10, 10)],
                [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            ),  # Median 10, IQR 0. Scaled to 0.
        ],
    )
    def test_robust_scaler_boundary(self, data, expected_scaled_data):
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data)
        assert len(scaled_data) == len(expected_scaled_data)
        for i in range(len(scaled_data)):
            assert len(scaled_data[i]) == len(expected_scaled_data[i])
            for j in range(len(scaled_data[i])):
                assert scaled_data[i][j] == pytest.approx(expected_scaled_data[i][j])

    def test_robust_scaler_fit_transform_separate(self):
        data = [(1, 2), (3, 4), (5, 6), (7, 8)]
        expected_scaled_data = [(-0.5, -0.5), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        scaler = RobustScaler()
        scaler.fit(data)
        scaled_data = scaler.transform(data)

        assert len(scaled_data) == len(expected_scaled_data)
        for i in range(len(scaled_data)):
            assert len(scaled_data[i]) == len(expected_scaled_data[i])
            for j in range(len(scaled_data[i])):
                assert scaled_data[i][j] == pytest.approx(expected_scaled_data[i][j])

    def test_robust_scaler_transform_before_fit_error(self):
        scaler = RobustScaler()
        data = [(1, 2), (3, 4)]
        with pytest.raises(NotFittedError):
            scaler.transform(data)
