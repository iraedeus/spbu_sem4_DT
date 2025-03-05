from math import sqrt

from ml_spbu.homework_2.annotations import Point
from ml_spbu.homework_2.exceptions import NotFittedError
from ml_spbu.homework_2.normalizers.abstract_scaler import AScaler


class StandardScaler(AScaler):
    def __init__(self):
        self._means = None
        self._deviations = None
        self._features_number = None

    def fit(self, X: list[Point]):
        self._features_number = len(X[0])
        n = self._features_number
        m = len(X)

        means = []
        deviations = []

        for i in range(n):
            mean_i = sum(x[i] for x in X) / m
            deviation_i = sqrt(sum((x[i] - mean_i) ** 2 for x in X) / m)

            means.append(mean_i)
            deviations.append(deviation_i)

        self._means = means
        self._deviations = deviations

    def transform(self, X: list[Point]):
        if not self._means or not self._deviations:
            raise NotFittedError()

        X_scaled = []
        n = self._features_number

        means = self._means
        deviations = self._deviations

        for x in X:
            x_scaled = []
            for i in range(n):
                if deviations[i] == 0:
                    x_scaled.append(0)
                else:
                    x_scaled.append((x[i] - means[i]) / deviations[i])
            X_scaled.append(tuple(x_scaled))

        return X_scaled

    def fit_transform(self, X: list[Point]):
        self.fit(X)
        return self.transform(X)
