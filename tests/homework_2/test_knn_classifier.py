import pytest

from ml_spbu.homework_2.KNNClassifier.knn_classifier import KNNClassifier
from ml_spbu.homework_2.annotations import Point
from ml_spbu.homework_2.KDTree.kd_tree import KDTree


def euclidean_distance(point1: Point, point2: Point) -> float:
    return sum([(x - y) ** 2 for x, y in zip(point1, point2)]) ** 0.5


class TestKNNClassifier:
    def test_init(self):
        knn_classifier = KNNClassifier(k=3, leaf_size=10, metric=euclidean_distance)
        assert knn_classifier._k == 3
        assert knn_classifier._leaf_size == 10
        assert knn_classifier._metric == euclidean_distance

    def test_fit(self):
        X = [(1, 2), (2, 3), (3, 4), (4, 5)]
        Y = [0, 1, 0, 1]
        knn_classifier = KNNClassifier(k=3, leaf_size=2, metric=euclidean_distance)
        knn_classifier.fit(X, Y)
        assert isinstance(knn_classifier._kd_tree, KDTree)
        assert knn_classifier._labels == {(1, 2): 0, (2, 3): 1, (3, 4): 0, (4, 5): 1}

    @pytest.mark.parametrize(
        "k, X_train, Y_train, X_predict, expected_proba",
        [
            (1, [(1, 2), (2, 3)], [0, 1], [(1.5, 2.5)], [(0.0, 1.0)]),
            (2, [(1, 2), (2, 3), (3, 4), (4, 5)], [0, 1, 0, 1], [(2.5, 3.5)], [(0.5, 0.5)]),
            (3, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], [0, 1, 0, 1, 0], [(3, 4)], [(1 / 3, 2 / 3)]),
            (4, [(1, 2), (2, 3), (3, 4), (4, 5)], [0, 1, 0, 1], [(3, 4)], [(0.5, 0.5)]),
            (1, [(0, 0), (0, 10), (10, 0), (10, 10)], [0, 0, 1, 1], [(1, 1)], [(1.0, 0.0)]),
        ],
    )
    def test_predict_proba_basic(self, k, X_train, Y_train, X_predict, expected_proba):
        knn_classifier = KNNClassifier(k=k, leaf_size=2, metric=euclidean_distance)
        knn_classifier.fit(X_train, Y_train)
        predicted_proba = knn_classifier.predict_proba(X_predict)
        assert pytest.approx(predicted_proba, abs=0.0001) == expected_proba

    @pytest.mark.parametrize(
        "k, X_train, Y_train, X_predict, expected_prediction",
        [
            (1, [(1, 2), (2, 3)], [0, 1], [(1.5, 2.5)], [1]),
            (2, [(1, 2), (2, 3), (3, 4), (4, 5)], [0, 1, 0, 1], [(2.5, 3.5)], [0]),
            (3, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], [0, 1, 0, 1, 0], [(3, 4)], [1]),
        ],
    )
    def test_predict_basic(self, k, X_train, Y_train, X_predict, expected_prediction):
        knn_classifier = KNNClassifier(k=k, leaf_size=2, metric=euclidean_distance)
        knn_classifier.fit(X_train, Y_train)
        predicted = knn_classifier.predict(X_predict)
        assert predicted == expected_prediction

    def test_predict_proba_before_fit(self):
        knn_classifier = KNNClassifier(k=3, leaf_size=2, metric=euclidean_distance)
        with pytest.raises(AttributeError):
            knn_classifier.predict_proba([(1, 1)])

    def test_predict_before_fit(self):
        knn_classifier = KNNClassifier(k=3, leaf_size=2, metric=euclidean_distance)
        X_predict = [(1, 1)]
        with pytest.raises(
            AttributeError
        ):  # Or perhaps a different error depending on implementation if _kd_tree is accessed before fit.
            knn_classifier.predict(X_predict)
