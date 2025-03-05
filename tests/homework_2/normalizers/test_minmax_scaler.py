import pytest

from ml_spbu.homework_2.exceptions import NotFittedError
from ml_spbu.homework_2.normalizers.minmax_scaler import MinMaxScaler


class TestMinMaxScaler:
    @pytest.mark.parametrize(
        "data, expected_scaled_data",
        [
            ([(1, 2), (3, 4), (5, 6)], [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]),
            ([(0, 0), (10, 10), (20, 20)], [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]),
            ([(-1, -1), (0, 0), (1, 1)], [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]),
            ([(1, 5), (2, 4), (3, 3)], [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]),
        ],
    )
    def test_min_max_scaler_fit_transform(self, data, expected_scaled_data):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        assert scaled_data == expected_scaled_data

    @pytest.mark.parametrize(
        "data, expected_scaled_data",
        [
            ([(5, 5)], [(0.0, 0.0)]),
            ([(0, 0), (0, 0), (0, 0)], [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]),
            ([(10, 10), (10, 10), (10, 10)], [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]),
        ],
    )
    def test_min_max_scaler_boundary(self, data, expected_scaled_data):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        assert scaled_data == expected_scaled_data

    def test_min_max_scaler_fit_transform_separate(self):
        data = [(1, 2), (3, 4), (5, 6)]
        expected_scaled_data = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        scaler = MinMaxScaler()
        scaler.fit(data)
        scaled_data = scaler.transform(data)
        assert scaled_data == expected_scaled_data

    def test_min_max_scaler_transform_before_fit_error(self):
        scaler = MinMaxScaler()
        data = [(1, 2), (3, 4)]
        with pytest.raises(NotFittedError):
            scaler.transform(data)
