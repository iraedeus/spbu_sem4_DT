import hypothesis.strategies as st
import pytest
from hypothesis import given
from sklearn.metrics import f1_score as sklearn_f1_score

from ml_spbu.homework_2.metrics.classification.f1_score import f1_score


@pytest.mark.parametrize("y_pred, y_true", [([0, 0, 0, 0], [0, 0, 0, 0])])
def test_f1_score_no_true_positives_and_false_negatives(y_pred, y_true):
    assert f1_score(y_pred, y_true) == 0.0


@pytest.mark.parametrize("y_pred, y_true", [([0, 0, 0, 0], [0, 1, 1, 0])])
def test_f1_score_no_true_and_false_positives(y_pred, y_true):
    assert f1_score(y_pred, y_true) == 0.0


@pytest.mark.parametrize("y_pred, y_true", [([1, 1, 0, 0], [0, 0, 0, 0])])
def test_f1_score_no_true_positives(y_pred, y_true):
    assert f1_score(y_pred, y_true) == 0.0


@given(data=st.data())
def test_f1_score_hypothesis(data):
    list_len = data.draw(st.integers(min_value=0, max_value=100))
    y_pred = data.draw(st.lists(st.integers(0, 1), min_size=list_len, max_size=list_len))
    y_true = data.draw(st.lists(st.integers(0, 1), min_size=list_len, max_size=list_len))

    calculated_f1 = f1_score(y_pred, y_true)
    sklearn_f1 = sklearn_f1_score(y_true, y_pred)

    assert pytest.approx(calculated_f1, rel=1e-6) == sklearn_f1


def test_f1_score_empty_lists():
    assert f1_score([], []) == 0.0


@pytest.mark.parametrize("y_pred, y_true", [([0, 1, 1, 1], [1, 1, 1, 1, 7])])
def test_f1_score_raises(y_pred, y_true):
    with pytest.raises(AttributeError):
        f1_score(y_pred, y_true)
