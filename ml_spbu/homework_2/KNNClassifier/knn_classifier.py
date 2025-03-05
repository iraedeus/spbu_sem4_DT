from ml_spbu.homework_2.annotations import CallableMetric, Point
from ml_spbu.homework_2.KDTree.kd_tree import KDTree


class KNNClassifier:
    def __init__(self, k: int, leaf_size: int, metric: CallableMetric):
        self._k = k
        self._leaf_size = leaf_size
        self._metric = metric

    def fit(self, X: list[Point], Y: list[int]):
        self._kd_tree = KDTree(X, self._leaf_size, self._metric)
        self._labels = {x: y for x, y in zip(X, Y)}

    def predict_proba(self, X: list[Point]) -> list[tuple[float, float]]:
        output = []
        results = self._kd_tree.query(X, self._k)

        for result in results:
            negatives = sum(self._labels[neighbor] == 0 for neighbor in result)
            positives = sum(self._labels[neighbor] == 1 for neighbor in result)
            output.append((negatives / len(result), positives / len(result)))

        return output

    def predict(self, X: list[Point]) -> list[int]:
        output = []
        labels = self.predict_proba(X)

        for proba in labels:
            output.append(int(proba[1] > proba[0]))

        return output
