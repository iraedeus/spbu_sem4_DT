import pytest
from hypothesis import given
import hypothesis.strategies as st

from ml_spbu.homework_2.train_test_split import train_test_split


class TestTrainTestSplit:
    @pytest.mark.parametrize(
        "X, y, test_size, random_state",
        [
            (
                [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)],
                [0, 1, 0, 1, 0],
                0.4,
                0,
            ),
            (
                [(1, 2), (3, 4), (5, 6)],
                [0, 1, 0],
                0.33,
                42,
            ),
            (
                [(1, 2)],
                [0],
                0.0,
                42,
            ),
            (
                [(1, 2)],
                [0],
                1.0,
                42,
            ),
        ],
    )
    def test_train_test_split_properties(self, X, y, test_size, random_state):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)

        assert pytest.approx(len(X_test) / len(X), abs=0.1) == test_size

        original_pairs = set(zip(X, y))
        train_pairs = set(zip(X_train, y_train))
        test_pairs = set(zip(X_test, y_test))

        assert train_pairs.union(test_pairs) == original_pairs
        assert not train_pairs.intersection(test_pairs)

    @pytest.mark.parametrize(
        "X, y, test_size, expected_X_train_len, expected_X_test_len, expected_y_train_len, expected_y_test_len",
        [
            ([], [], 0.5, 0, 0, 0, 0),
            ([(1, 2)], [0], 0.0, 1, 0, 1, 0),
            ([(1, 2)], [0], 1.0, 0, 1, 0, 1),
            ([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)], [0, 1, 0, 1, 0], 0.0, 5, 0, 5, 0),
            ([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)], [0, 1, 0, 1, 0], 1.0, 0, 5, 0, 5),
            ([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)], [0, 1, 0, 1, 0], 0.5, 3, 2, 3, 2),
            ([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)], [0, 1, 0, 1, 0, 1], 0.3, 4, 2, 4, 2),
        ],
    )
    def test_train_test_split_lengths(
        self, X, y, test_size, expected_X_train_len, expected_X_test_len, expected_y_train_len, expected_y_test_len
    ):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        assert len(X_train) == expected_X_train_len
        assert len(X_test) == expected_X_test_len
        assert len(y_train) == expected_y_train_len
        assert len(y_test) == expected_y_test_len

    @pytest.mark.parametrize(
        "X, y, test_size, random_state",
        [
            ([(1, 2), (3, 4)], [0, 1], 2.0, 42),
            ([(1, 2), (3, 4)], [0, 1], -0.1, 42),
        ],
    )
    def test_train_test_split_invalid_test_size(self, X, y, test_size, random_state):
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=test_size, random_state=random_state)

    @given(test_size=st.floats(min_value=0.0, max_value=1.0), random_state=st.integers())
    def test_train_test_split_sum_lengths_and_ratio(self, test_size, random_state):
        X = [(i, i + 1) for i in range(100)]
        y = list(range(100))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert pytest.approx(len(X_test) / len(X), abs=0.1) == test_size

        original_pairs = set(zip(X, y))
        train_pairs = set(zip(X_train, y_train))
        test_pairs = set(zip(X_test, y_test))

        assert train_pairs.union(test_pairs) == original_pairs
        assert not train_pairs.intersection(test_pairs)

    @given(random_state=st.integers())
    def test_train_test_split_deterministic(self, random_state):
        X = [(i, i + 1) for i in range(50)]
        y = list(range(50))
        test_size = 0.3
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=test_size, random_state=random_state)
        assert X_train1 == X_train2
        assert X_test1 == X_test2
        assert y_train1 == y_train2
        assert y_test1 == y_test2
