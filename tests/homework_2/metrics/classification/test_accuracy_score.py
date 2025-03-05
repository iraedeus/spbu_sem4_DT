import pytest

from ml_spbu.homework_2.metrics.classification.accuracy_score import accuracy_score


@pytest.mark.parametrize(
    "y_pred, y_true, expected",
    [
        ([0, 1, 1, 0, 1], [0, 1, 1, 0, 1], 1.0),
        ([0, 1, 1, 0, 1], [1, 0, 0, 1, 0], 0),
        ([1, 1, 1, 1, 1], [0, 0, 0, 0, 0], 0),
        ([0, 1, 1, 0, 1], [0, 0, 1, 1, 1], 0.6),
    ],
)
def test_accuracy_score_valid_input(y_pred, y_true, expected):
    assert accuracy_score(y_pred, y_true) == expected


def test_accuracy_score_empty_lists():
    assert accuracy_score([], []) == 1.0


@pytest.mark.parametrize("y_pred, y_true", [([0, 1, 1, 1], [1, 1, 1, 1, 7])])
def test_accuracy_score_raises(y_pred, y_true):
    with pytest.raises(AttributeError):
        accuracy_score(y_pred, y_true)
