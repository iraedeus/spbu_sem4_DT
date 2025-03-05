import pytest

from ml_spbu.homework_2.exceptions import NotFittedError
from ml_spbu.homework_2.normalizers.standard_scaler import StandardScaler


class TestStandardScaler:
    @pytest.mark.parametrize(
        "data, expected_scaled_data",
        [
            ([(1, 2), (3, 4), (5, 6)], [(-1.224, -1.224), (0.0, 0.0), (1.224, 1.224)]),
            ([(0, 0), (2, 2), (4, 4)], [(-1.224, -1.224), (0.0, 0.0), (1.224, 1.224)]),
        ],
    )
    def test_standard_scaler_basic(self, data, expected_scaled_data):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        assert len(scaled_data) == len(expected_scaled_data)
        for i in range(len(scaled_data)):
            assert len(scaled_data[i]) == len(expected_scaled_data[i])
            for j in range(len(scaled_data[i])):
                assert scaled_data[i][j] == pytest.approx(expected_scaled_data[i][j], rel=1e-3)

    @pytest.mark.parametrize(
        "data, expected_scaled_data",
        [
            ([(5, 5)], [(0.0, 0.0)]),
            ([(0, 0), (0, 0), (0, 0)], [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]),
            ([(10, 10), (10, 10), (10, 10)], [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]),
        ],
    )
    def test_standard_scaler_boundary(self, data, expected_scaled_data):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        assert len(scaled_data) == len(expected_scaled_data)
        for i in range(len(scaled_data)):
            assert len(scaled_data[i]) == len(expected_scaled_data[i])
            for j in range(len(scaled_data[i])):
                assert scaled_data[i][j] == pytest.approx(expected_scaled_data[i][j])

    def test_standard_scaler_fit_transform_separate(self):
        data = [(1, 2), (3, 4), (5, 6)]
        expected_scaled_data = [(-1.224, -1.224), (0.0, 0.0), (1.224, 1.224)]
        scaler = StandardScaler()
        scaler.fit(data)
        scaled_data = scaler.transform(data)
        assert len(scaled_data) == len(expected_scaled_data)
        for i in range(len(scaled_data)):
            assert len(scaled_data[i]) == len(expected_scaled_data[i])
            for j in range(len(scaled_data[i])):
                assert scaled_data[i][j] == pytest.approx(expected_scaled_data[i][j], rel=1e-3)

    def test_standard_scaler_transform_before_fit_error(self):
        scaler = StandardScaler()
        data = [(1, 2), (3, 4)]
        with pytest.raises(NotFittedError):
            scaler.transform(data)
